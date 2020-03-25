from dlutils.models.gans.adversarial_autoencoder import \
    AdversarialAutoEncoderPyTorch
from dlutils.models.gans.auxiliary_classifier import \
    AuxiliaryClassifierGANPyTorch
from dlutils.models.gans.boundary_equilibrium import BoundaryEquilibriumGAN
from dlutils.models.gans.boundary_seeking import BoundarySeekingGAN
from dlutils.models.gans.conditional import ConditionalGAN
from dlutils.models.gans.context_conditional import ContextConditionalGAN
from dlutils.models.gans.context_encoder import ContextEncoder
from dlutils.models.gans.coupled import CoupledGAN
from dlutils.models.gans.cycle import CycleGAN
from dlutils.models.gans.deep_convolutional import DeepConvolutionalGAN
from dlutils.models.gans.disco import DiscoGAN
from dlutils.models.gans.dragan import DRAGAN
from dlutils.models.gans.dual import DualGAN
from dlutils.models.gans.energy_based import EnergyBasedGAN
from dlutils.models.gans.enhanced_super_resolution import \
    EnhancedSuperResolutionGAN
from dlutils.models.gans.gan import GenerativeAdversarialNetworks
from dlutils.models.gans.info import InfoGAN
from dlutils.models.gans.munit import MUNIT
from dlutils.models.gans.pix2pix import Pix2Pix
from dlutils.models.gans.pixel_da import PixelDomainAdaptation
from dlutils.models.gans.relativistic import RelativisticGAN
from dlutils.models.gans.semi_supervised import SemiSupervisedGAN
from dlutils.models.gans.softmax import SoftmaxGAN
from dlutils.models.gans.star import StarGAN
from dlutils.models.gans.super_resolution import SuperResolutionGAN
from dlutils.models.gans.unit import UNIT
from dlutils.models.gans.wasserstein import WassersteinGAN
from dlutils.models.gans.wasserstein_div import WassersteinDivergenceGAN
from dlutils.models.gans.wasserstein_gp import WassersteinGradientPenaltyGAN

# make LSGAN a synonym for basic GAN, since training only differs in loss
# function, which isn't specified here
LeastSquareGAN = GenerativeAdversarialNetworks
