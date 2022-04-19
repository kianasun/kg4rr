from typing import Any, ClassVar, Mapping
from pykeen.nn.combinations import ComplExLiteralCombination

import torch.autograd
from torch import linalg
from torch.nn import functional
import torch
import sys

from pykeen.models.nbase import EmbeddingSpecificationHint
from pykeen.models.base import EntityRelationEmbeddingModel
from pykeen.utils import complex_normalize
from pykeen.constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,\
                                DEFAULT_DROPOUT_HPO_RANGE
from pykeen.losses import BCEWithLogitsLoss
from pykeen.nn.emb import EmbeddingSpecification
from pykeen.nn.init import xavier_uniform_, xavier_uniform_norm_, init_phases
from pykeen.nn.modules import ComplExInteraction, FunctionalInteraction, LiteralInteraction
from pykeen.nn.combinations import Combination
from pykeen.typing import Constrainer, Hint, Initializer
#from utils import PLLiteralModel, TriplesLiteralsFactory
from pykeen.models.nbase import EmbeddingSpecificationHint, ERModel
from pykeen.typing import HeadRepresentation, RelationRepresentation, \
                        TailRepresentation
from pykeen.nn.emb import Embedding, LiteralRepresentation
from pykeen.triples import TriplesFactory

class PretrainedInitializer:

    def __init__(self, tensor: torch.FloatTensor) -> None:
        self.tensor = tensor.clone()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize the tensor with the given tensor."""
        if x.shape != self.tensor.shape:
            raise ValueError(f"shape does not match: expected {self.tensor.shape} but got {x.shape}")
        return self.tensor.to(device=x.device, dtype=x.dtype)

class TriplesLiteralsFactory(TriplesFactory):
    """Create multi-modal instances given the path to triples."""

    def __init__(
        self,
        *,
        triples=None,
        literal_embedding=None,
        **kwargs,
    ) -> None:
        base = TriplesFactory(mapped_triples=triples, **kwargs)
        super().__init__(
            entity_to_id=base.entity_to_id,
            relation_to_id=base.relation_to_id,
            mapped_triples=base.mapped_triples,
            create_inverse_triples=base.create_inverse_triples,
        )

        self.literal_embedding = literal_embedding
        #print("Literal Embedding shape: {}".format(literal_embedding.shape))
        #print("In triplefactory", self.literal_embedding[:2, :5])

    def get_literals_tensor(self) -> torch.FloatTensor:
        """Return the numeric literals as a tensor."""
        return torch.as_tensor(self.literal_embedding, dtype=torch.float32)

class PLLiteralModel(ERModel[HeadRepresentation, RelationRepresentation, TailRepresentation], autoreset=False):
    """Base class for models with entity literals that uses combinations from :class:`pykeen.nn.combinations`."""

    def __init__(
        self,
        triples_factory: TriplesLiteralsFactory,
        interaction,
        entity_representations,
        relation_representations: EmbeddingSpecificationHint = None,
        **kwargs,
    ):
        literals = triples_factory.get_literals_tensor()
        num_embeddings, shape = literals.shape
        print("Literal Embedding shape for initializing: {}".format(literals.shape))
        print('Sample vector ', literals[0, :10])

        literal_representation = Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=shape,
            initializer=PretrainedInitializer(tensor=literals),
            trainable=False,
        )

        #print(literal_representation._embeddings.shape)
        literal_representation._embeddings.weight[:, :] = literals
        #print(literal_representation._embeddings.weight.dtype)
        #print(literal_representation._embeddings.weight.shape)
        #print(literal_representation._embeddings.weight[:2, :5])
        literal_representation._embeddings.requires_grad_(False)
        #print(literal_representation._embeddings.requires_grad_())
        """
        literal_representation = Embedding(
            num_embeddings=num_embeddings,
            shape=shape
        )

        #literal_representation._embeddings.weight = torch.nn.Parameter(literals)
        literal_representation._embeddings.weight.data = literals
        literal_representation._embeddings.requires_grad_ = False
        print("initialized literal representation", \
                    literal_representation(torch.LongTensor([0, 1]))[:, :5])
        print("initialized literal directly", \
                    literal_representation._embeddings(torch.LongTensor([0, 1]))[:, :5])
        #print("passed in", literals[:2, :5])
        """
        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            entity_representations=[*entity_representations, literal_representation],
            relation_representations=relation_representations,
            **kwargs,
        )


class CombineLiteral(PLLiteralModel):
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        input_dropout=DEFAULT_DROPOUT_HPO_RANGE,
    )
    #: The default loss function class
    loss_default = BCEWithLogitsLoss
    #: The default parameters for the default loss function class
    loss_default_kwargs: ClassVar[Mapping[str, Any]] = {}

    def __init__(
        self,
        triples_factory: TriplesLiteralsFactory,
        embedding_dim: int = 50,
        input_dropout: float = 0.2,
        base_interaction: FunctionalInteraction=ComplExInteraction,
        combine_interaction: Combination=ComplExLiteralCombination,
        **kwargs,
    ) -> None:
        """Initialize the model."""
        base_class_name = base_interaction.__name__
        #print("Base Interaction Class", base_class_name)
        interact_kwargs = kwargs["interact_kwargs"]
        del kwargs["interact_kwargs"]
        if "ComplEx" in base_class_name or "RotatE" in base_class_name:
            base_type = torch.cfloat
        else:
            base_type = torch.float
        super().__init__(
            triples_factory=triples_factory,
            interaction=LiteralInteraction(
                base=base_interaction(**interact_kwargs),
                combination=combine_interaction(
                    entity_embedding_dim=embedding_dim,
                    literal_embedding_dim=triples_factory.literal_embedding.shape[-1],
                    input_dropout=input_dropout,
                ),
            ),
            entity_representations=[
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=torch.nn.init.xavier_normal_,
                    dtype=base_type
                ),
            ],
            relation_representations=[
                EmbeddingSpecification(
                    embedding_dim=embedding_dim,
                    initializer=torch.nn.init.xavier_normal_,
                    dtype=base_type
                ),
            ],
            **kwargs,
        )
        #print(self.interaction.combination.literal_embedding)


class EvaluateModel(torch.nn.Module):
    def __init__(self, base_model, content_emb=None, num_emb=None, model_name=None, eval_type=None):
        super(EvaluateModel, self).__init__()

        base_model.eval()
        self.content_emb = content_emb
        self.num_emb = num_emb

        self.eval_type=eval_type

        self.model_name = model_name
        self.dim_reduction = None

        if "Combine" in self.model_name:
            _node = base_model.entity_representations[0](indices=None).detach()
            _desc = base_model.entity_representations[1](indices=None).detach()
            self.ent_emb = base_model.interaction.combination(_node, _desc).detach()
        else:
            self.ent_emb = base_model.entity_representations[0](indices=None).detach()
        print("Embedding shape", self.ent_emb.shape, content_emb.shape)

        self.entity_dim = self.ent_emb.shape[1]

        mid_dim = int(self.entity_dim / 2)

        if eval_type == "node":
            self.cls_head = torch.nn.Sequential(
                torch.nn.Linear(self.entity_dim, mid_dim),
                torch.nn.LeakyReLU(0.2),
                torch.nn.LayerNorm(mid_dim),
                torch.nn.Linear(mid_dim, 1),
                torch.nn.Sigmoid()
            )
        elif eval_type == "content":
            assert self.content_emb is not None
            self.cls_head = torch.nn.Sequential(
                torch.nn.Linear(self.content_emb.shape[1], mid_dim),
                torch.nn.LeakyReLU(0.2),
                torch.nn.LayerNorm(mid_dim),
                torch.nn.Linear(mid_dim, 1),
                torch.nn.Sigmoid()
            )
        elif eval_type == "number":
            assert self.num_emb is not None
            _shape = self.num_emb.shape[1]
            self.cls_head = torch.nn.Sequential(
                torch.nn.Linear(_shape, mid_dim),
                torch.nn.LeakyReLU(0.2),
                torch.nn.LayerNorm(mid_dim),
                torch.nn.Linear(mid_dim, 1),
                torch.nn.Sigmoid()
            )
        elif eval_type == "all":
            assert (self.num_emb is not None) and (self.content_emb is not None)
            _shape = self.content_emb.shape[1] + self.num_emb.shape[1]
            self.cls_head = torch.nn.Sequential(
                torch.nn.Linear(self.entity_dim+_shape, mid_dim),
                torch.nn.LeakyReLU(0.2),
                torch.nn.LayerNorm(mid_dim),
                torch.nn.Linear(mid_dim, 1),
                torch.nn.Sigmoid()
            )

        self.cls_loss = torch.nn.MSELoss()

    def forward(self, indices, scores):

        node_emb = torch.index_select(
            self.ent_emb,
            dim=0,
            index=indices
        )

        text_emb = torch.index_select(
            self.content_emb,
            dim=0,
            index=indices
        )

        num_emb = torch.index_select(
            self.num_emb,
            dim=0,
            index=indices
        )

        if self.eval_type == "node":
            ent_scores = self.cls_head(node_emb)
        elif self.eval_type == "content":
            ent_scores = self.cls_head(text_emb)
        elif self.eval_type == "number":
            ent_scores = self.cls_head(num_emb)
        elif self.eval_type == "all":
            ent_scores = self.cls_head(torch.cat((node_emb, text_emb, num_emb), dim=1))
        else:
            print("Error")
            sys.exit(1)

        ent_scores = ent_scores.squeeze(1)
        if scores is None:
            return [_.item() for _ in ent_scores]
        else:
            loss = self.cls_loss(ent_scores, scores)
            return ent_scores.cpu().detach().numpy().tolist(), loss
