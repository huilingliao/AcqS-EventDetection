import torch
import warnings

import numpy as np
from torch import Tensor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition.analytic import *
from botorch.acquisition.acquisition import AcquisitionFunction, OneShotAcquisitionFunction
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.utils import is_nonnegative

from botorch.optim.utils import columnwise_clamp
from botorch.optim.initializers import sample_points_around_best, initialize_q_batch_nonneg, initialize_q_batch

from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.utils.sampling import draw_sobol_samples

from botorch.logging import logger

from botorch.test_functions.synthetic import * 
from test_functions_add import *

from HMC_pytorch.hmc import HMC, HMC_wc, unique
from scipy.stats.qmc import LatinHypercube

r"""
Data initialization utilities for synthetic test functions. 
"""

def initialize_func_data(n_init, test_fname, noise_err = None, negate = True):
    r""" Generate initial data X and Y 
    
    Args: 
        n_init: The number of initial observations
        test_func: The synthetic test function
    """

    if test_fname == "forrester":
        test_func = Forrester(noise_std = noise_err, negate = negate)
        X = torch.tensor([[1.000000], [0.998999], [0.997998], [0.996997], [0.995996], [0.994995],
                          [0.993994], [0.992993], [0.991992], [0.990991], [0.989990], [0.988989],
                          [0.987988], [0.986987], [0.985986], [0.984985], [0.983984], [0.982983],
                          [0.981982], [0.000001]])
        
    elif test_fname == "gramacylee":
        test_func = Gramacylee(noise_std = noise_err, negate = negate)
        X = torch.tensor([[2.500000], [2.479798], [2.459596], [2.439394], [2.419192], [2.398990],
                          [2.378788], [2.358586], [2.338384], [2.318182], [2.297980], [2.277778],
                          [2.257576], [2.237374], [2.217172], [2.196970], [2.176768], [2.156566],
                          [2.075758], [2.055556]])

    elif test_fname == "branin":
        test_func = Branin(noise_std = noise_err, negate = negate)
        X = torch.tensor([[-4.164435,  1.1083613], [ 6.528743, 14.0159310], [-4.698525,  3.2222688],
                          [ 7.539784, 12.6726376], [-3.259308,  0.9139629], [ 9.412842, 13.9978708],
                          [ 3.461396, 13.0616971], [ 9.995536, 13.8777384], [-3.106979,  2.5912381],
                          [ 4.747401, 10.0837920], [ 6.207700,  8.9837226], [ 7.221728,  9.3089992],
                          [ 4.267313,  9.9415172], [ 6.231150,  8.5540955], [-3.482933,  4.5695854],
                          [ 7.683778,  9.2480085], [-2.705795,  3.0076102], [ 7.992003,  9.3048741],
                          [ 7.421597,  7.8761192], [ 9.984521, 10.3204020]])

    elif test_fname == "bukin":
        test_func = Bukin(noise_std = noise_err, negate = negate)
        X = torch.tensor([[-14.782258, -2.31354514], [-11.733797, -2.73409899], [-12.577094, -1.73905221],
                          [ -5.551925, -2.95893695], [-14.752655, -0.83328976], [ -7.173028, -2.33736603],
                          [ -6.661439,  2.85186984], [-10.187024,  2.89301550], [ -5.942858,  2.02771586],
                          [-12.306895,  0.22623898], [-10.133092,  1.93559975], [-13.658440,  1.08699211],
                          [-12.114897,  2.08728042], [-13.417095, -0.97136246], [-11.839337, -1.85326349],
                          [ -8.973642, -1.92071465], [ -9.893225, -1.16856178], [ -8.735384, -0.81759850],
                          [ -7.044243,  2.19077014], [-12.719510, -0.81805064]])

    elif test_fname == "cos25":
        test_func = Cos25(noise_std = noise_err, negate = negate)
        X = torch.tensor([[  9.685261,  -9.453546], [-2.924930, -15.560466], [-9.287374, -15.484687],
                          [ -3.318463,   9.221838], [ 3.358412,   9.170040], [ 3.242191,   9.090878],
                          [ -9.638393,  15.379800], [ 9.363891,   9.027288], [-9.518078,   9.909692],
                          [ -9.358408,  -9.963629], [-9.294521,   2.590039], [ 3.676199,  -9.632932],
                          [  2.555179, -15.643330], [ 3.285424,  -2.558092], [10.025550,  -9.555165],
                          [-15.477495,  -8.854642], [ 3.448055,   3.684097], [ 8.852610,  15.435707],
                          [  9.438801,  -2.476219], [-9.449135,   3.826313]])
    
    elif test_fname == "sixhump":
        test_func = SixHumpCamel(noise_std = noise_err, negate = negate)
        X = torch.tensor([[-2.98328587, -1.572829267], [ 2.93459861,  1.626985202],
                          [ 2.93197648,  1.515288445], [ 2.76619446,    1.8223747],
                          [ 2.91231241, -1.984317077], [-0.46328674,  -1.99242722],
                          [ 0.13755351,   1.99900954], [-1.54981579, -1.996775675],
                          [ 0.55991782, -1.809796094], [-2.93046112,  -0.89865011],
                          [-2.90413808,   0.47318327], [ 0.87844217,   1.92421472],
                          [ 1.35101969,  -1.59447605], [ 2.91540293,   0.95101786],
                          [ 1.27818697,  -1.99910239], [ 2.94381701,   1.21146150],
                          [-2.93594566,  -0.25421254], [ 1.62054871,   1.99687208],
                          [ 1.31344912,   1.92525670], [ 0.78236839,   1.86286291]])

    elif test_fname == "ackley_6":
        test_func = Ackley(dim = 6, noise_std = noise_err, negate = negate)
        X = torch.tensor([[  -8.4315812, -17.3153522,  22.26262,  32.408878, -30.8963212,  -8.515600],
                          [  29.5453963,   1.6242493,  17.47136, -26.882310,  19.3924436,   8.618380],
                          [ -27.3038439, -28.1785901, -31.16398, -27.615676,  -8.5999745, -29.182121],
                          [ -25.5016057, -19.3620366, -22.09387,  32.053522,  29.8008629,  32.603029],
                          [  21.4021348, -21.2699596, -14.96034, -32.372521, -28.6480383, -27.973872],
                          [  20.4039755, -23.3293251, -22.49763,  24.721336,  12.2968080,  -9.095862], 
                          [  21.7826166,   0.8571889, -30.78154, -22.708268,   1.5109326, -28.386300],
                          [  24.9551382,   2.9504992,  12.53158, -11.583200,  24.6215372, -29.422976],
                          [  16.4992870,   6.7816827, -28.54615, -29.139715, -17.0625433,  15.599798],
                          [ -17.1774412,  21.4889229, -19.78724,  24.435184, -17.2409164,   6.550005],
                          [ -31.4464921, -32.1108239,  27.96421,   3.806778, -28.9037047,  28.537022],
                          [ -26.1758454,  18.1160973,  26.23840,  -8.599178, -31.0760322, -16.431081],
                          [  -3.8424573,  21.5174936, -23.57530,   3.151505,  -8.3470128, -26.564979],
                          [ -27.3098929, -31.1795910, -28.84553,  -8.799973, -26.9299267,   7.576955],
                          [  13.7991309,  -9.6431461, -14.43782, -28.950492,  -0.3213235,  32.162446],
                          [  30.0414338,  21.9310283, -20.42654, -19.781156,  19.8889582,  31.633125],
                          [ -15.3538295, -14.0652300, -12.43462,  19.034870,  28.7355148,  28.836614],
                          [   3.8419054, -11.0712230, -28.43576, -32.166735,  -5.1937491, -13.465382],
                          [   0.3696409, -23.1028801,  20.13654,  22.470700, -12.3875389,  -2.613469],
                          [  31.6613292,  32.4172927, -16.05224,  16.105771,  -6.8701088,  -5.216815]])

    elif test_fname == "ackley_10":
        test_func = Ackley(dim = 10, noise_std = noise_err, negate = negate)
        X = torch.tensor([[ 32.3877971,  26.501815,  22.299357, -26.040931, -30.281209,  21.511766, -30.6450437,   3.3848704,  -3.747893,   8.421723],
                          [ -0.4018215,  28.605290, -32.462399, -31.150627,   2.292289, -30.278813, -10.5155881, -29.1811123,  -8.628492,  22.863357],
                          [  8.3869819,  19.429193, -13.957686,  -5.509045, -31.332671, -32.226475,  -2.1937338,  25.5622947,  16.517047, -23.703583],
                          [-28.0666682,  21.768969,  26.401945, -22.478409,   3.790090,  -6.285940, -14.2455309, -31.1224398,  32.593790,  19.583546],
                          [ 28.1266138, -30.221196,   5.582365,  28.450221,   8.392860,   4.626442,  21.8653813,  17.3430815,  18.442893, -21.848996],
                          [-23.8984996,  22.107031,  23.799612,  -9.613451,  -2.312701,  32.471670,   7.4874878,  12.7587307,  -5.315025,  31.492374],
                          [  4.4944473, -22.318968, -30.107162,  -2.360623,  23.108870, -32.179005, -29.8295934,   3.5907822,  -2.515658,   9.514477],
                          [  3.2535941,  29.247259,  -4.555638, -15.568783, -17.402359,  32.761535, -26.9948812,  23.5434795,   5.071997,  -3.649046],
                          [-26.4332415, -11.818434,  13.259183, -16.497368,  24.822449,  28.089611,  10.5784070, -23.0773747, -24.634559,   2.415521],
                          [ 12.0220716, -23.303859,  22.247031, -32.295907,  21.254176,  -8.062512, -31.7556683,  23.6022372, -28.132989, -29.306295],
                          [ 28.4457317, -31.900993,  11.725527,  13.701073,  16.253916,  17.180442,  23.5799218, -16.0500937, -14.822065, -26.369948],
                          [ 22.6049354,   3.144253,  -3.647890,  15.606272,  22.827076,  15.700359,   0.2790672, -26.2114408,  32.471867,  -8.332550],
                          [  2.8112157, -30.208566, -28.262294,  23.307709,   6.985436, -21.499411,  30.4090958, -14.0798504, -29.016723,  -6.497402],
                          [-29.0509108, -31.388581,  -1.786883,  19.682283,  21.778707,  25.730137,   5.7497075,  -0.7247546, -22.874676,  17.616220],
                          [-30.7877591,   6.088867,  -8.743845,  28.313675, -15.652337, -30.404829, -13.9825939, -19.1323187, -17.318556,  23.295580],
                          [-28.9772533, -12.650350, -30.481810, -24.112105, -21.858462,  12.936587, -19.3420499,  14.1864457, -11.306360, -28.364213],
                          [ -2.7898445, -27.741687, -18.224266,  -1.718605, -12.544992,  29.952344, -17.9233362,  21.5872144, -14.626348,  28.231862],
                          [ 32.2619098, -32.080755,  32.409694,  -4.724463, -27.900768, -10.824707,  -1.0797286,  -6.3286346,  -7.783757, -19.469783],
                          [ 12.0866209,  -1.431096,  23.915167, -31.290300, -24.881174,  12.410466,  24.9632130,  23.9311524,  29.463442,   7.787244],
                          [ 30.0908081,   8.790618,  -5.752885,   7.907234, -28.152526,  26.106973,  28.5196678, -32.7164891,  11.628945,  23.099142]])

    elif test_fname == "hartmann":
        test_func = Hartmann(noise_std = noise_err, negate = negate)
        X = torch.tensor([[0.917491171, 0.31974692224, 0.8519555288, 0.2657330546, 0.957700116, 0.482562402],
                          [0.770659772, 0.11653138814, 0.9827676432, 0.7120511250, 0.862200331, 0.824269375],
                          [0.957141051, 0.96626577619, 0.0014717572, 0.8152684912, 0.426642967, 0.375274350],
                          [0.886041189, 0.14163601818, 0.6169752553, 0.3130856985, 0.882597998, 0.889040401],
                          [0.995713057, 0.19697296061, 0.3936798437, 0.8782727711, 0.525233528, 0.422771374],
                          [0.117185918, 0.14694812358, 0.2073540508, 0.9181328288, 0.854374080, 0.832469278],
                          [0.855443630, 0.73070535786, 0.0233771447, 0.7360681910, 0.014971097, 0.615451148],
                          [0.879061422, 0.09180245525, 0.8522684642, 0.3053059964, 0.761649901, 0.404634016],
                          [0.236442405, 0.53545816569, 0.3102159053, 0.9875037423, 0.357519740, 0.995389137],
                          [0.864053558, 0.13308687159, 0.3983544284, 0.4906786461, 0.775472443, 0.509765224],
                          [0.554277961, 0.10227611987, 0.1862053028, 0.7850359816, 0.191730443, 0.015518636],
                          [0.442280527, 0.65666235588, 0.7902077872, 0.1150079516, 0.849898158, 0.992224550],
                          [0.657576123, 0.25019891234, 0.7694613419, 0.7709350730, 0.514761515, 0.274110735],
                          [0.235498313, 0.03543934086, 0.5375724582, 0.7697524345, 0.903988519, 0.965331213],
                          [0.793743405, 0.98793104268, 0.1767423782, 0.7050258562, 0.688176861, 0.370083976],
                          [0.823678245, 0.66690863087, 0.8832108213, 0.2357671564, 0.012945082, 0.409940149],
                          [0.337776616, 0.18705414305, 0.3137723736, 0.7475840838, 0.029630197, 0.114214398],
                          [0.023633718, 0.21494590305, 0.3058558111, 0.6690362883, 0.864621475, 0.495771657],
                          [0.861791465, 0.05977980117, 0.1290991304, 0.4225857565, 0.633873692, 0.421955053],
                          [0.582552165, 0.45579134882, 0.6629171574, 0.1349195810, 0.773142474, 0.300817263]])

    elif test_fname == "michalewicz_2":
        test_func = Michalewicz(dim = 2, noise_std = noise_err, negate = negate)
        X = torch.tensor([[0.087826736,  0.069116402], [0.012817273,  0.163849103], [0.063912353,  0.127418801], [0.186616258,  0.035294068],
                          [0.013336331,  0.238665764], [ 3.11981132, 0.2261725713], [3.078576702,  0.281922065], [3.112934513,  2.267103644],
                          [3.046981897,  3.101754896], [ 0.52718394, 0.2348220495], [ 0.60564961, 0.4299477234], [0.380039916,  0.061250420],
                          [0.286234941,  0.391561785], [0.844867621,  2.377461780], [1.150224611,  2.985051407], [1.404788955,  3.124873028],
                          [1.188944177,  1.982791676], [1.419435594,  0.189363624], [2.781520844,  2.430745204], [1.465455198,  2.320642085]])

    elif test_fname == "michalewicz_5":
        test_func = Michalewicz(dim = 5, noise_std = noise_err, negate = negate)
        X = torch.tensor([[0.787318919, 0.491606242, 0.298278284, 2.1974807450, 2.4119346578],
                          [0.721429263, 2.212243160, 2.637477647, 0.0683040679, 0.4677493912],
                          [2.862951550, 2.365068917, 0.091228582, 1.5165600902, 1.3381990304],
                          [1.045920222, 3.073538290, 0.478391403, 2.1202814527, 3.1136793845],
                          [0.935013315, 0.157242448, 0.833762636, 0.1005464444, 3.1379405532],
                          [1.205626397, 1.993811891, 1.982791119, 2.8160636157, 0.1530387971],
                          [0.496999140, 0.871131496, 1.751056055, 0.0873600512, 0.6793332145],
                          [2.755095777, 2.106297757, 0.296810129, 3.0399481077, 0.3236964075],
                          [2.556405316, 1.958477190, 0.557985346, 1.5578931962, 3.0608311953],
                          [2.752306461, 2.570281799, 0.262265440, 0.4440758386, 0.2006607933],
                          [0.715304046, 1.819724148, 1.552842628, 3.0466659226, 0.2023830948],
                          [2.975096518, 0.589005422, 0.540205807, 2.7265113883, 1.1301830361],
                          [0.628304953, 1.789597707, 1.731257797, 0.5290244296, 0.3354537324],
                          [0.787717892, 1.101795318, 1.834661012, 1.3508852570, 1.6384384382],
                          [0.039648077, 0.211846406, 0.295728774, 3.1346705224, 0.8647835768],
                          [0.675738180, 1.393184304, 2.434587546, 3.1034714201, 2.5551569622],
                          [1.560820565, 1.051741177, 1.423213938, 1.4488404566, 2.5762628236],
                          [0.409111599, 0.735620955, 0.362685025, 0.1765729416, 2.6117614115],
                          [0.194047216, 0.263685426, 1.575667007, 3.0043694090, 2.1862486838],
                          [2.348568122, 0.264687116, 2.565045098, 1.6332440330, 0.1695581587]])


    elif test_fname == "michalewicz_10":
        test_func = Michalewicz(dim = 10, noise_std = noise_err, negate = negate)
        X = torch.tensor([[2.617447727, 3.065329431, 2.422405869, 3.036980923, 1.402111305, 1.118501971, 0.590740389, 1.945707385, 0.96484845, 3.131331040],
                          [1.238596320, 2.130405745, 0.072068114, 1.738235515, 1.601412122, 2.918258999, 0.020515675, 3.124203006, 0.44892957, 2.015495388],
                          [2.634452221, 1.020393286, 2.105202820, 2.757223125, 0.389833004, 0.027629861, 0.584826389, 0.297620412, 3.04685696, 1.680012507],
                          [0.286469860, 2.418207800, 0.867314255, 2.890473724, 3.062866275, 1.426638774, 0.410860987, 3.038424612, 0.29145008, 0.503104666],
                          [0.041531069, 3.136724023, 0.534231370, 2.038309846, 3.136906566, 0.331969503, 2.806080290, 0.076829859, 2.94112890, 2.748253870],
                          [1.438272709, 1.179511654, 1.583867126, 1.620336247, 1.395131895, 2.092237328, 1.930954834, 3.032892612, 3.12704704, 0.591447568],
                          [1.791828823, 2.138841320, 3.012097407, 0.612641434, 1.109302539, 1.810821762, 1.131879570, 0.404351924, 1.37570001, 0.790546113],
                          [1.580005179, 0.479237862, 0.613474092, 0.046072891, 1.083562504, 1.482787103, 1.948714906, 0.555948022, 1.40153638, 0.895133979],
                          [2.894986154, 2.996735589, 2.463563421, 2.818037508, 2.004778521, 1.033847395, 1.511920760, 2.852389332, 1.01691432, 0.557900942],
                          [2.982851764, 1.880133667, 0.597607647, 2.518890307, 2.984647306, 0.326556418, 1.625635481, 1.580805077, 1.15784618, 1.924894032],
                          [1.387413831, 1.864921908, 1.019784121, 0.393237772, 2.864333259, 1.998834075, 1.112369671, 0.198879757, 1.83747667, 1.134125607],
                          [0.731710602, 2.001500634, 0.183262094, 2.217542992, 1.082039463, 1.744603837, 0.085006682, 2.043698072, 2.09844468, 0.598041687],
                          [1.253923632, 0.062007563, 1.577969221, 1.171732615, 0.796679718, 2.951514330, 1.255112423, 2.300968096, 1.41627476, 1.801682808],
                          [3.003156565, 0.185237569, 2.978221573, 1.796788637, 0.924301433, 1.040094736, 2.883730702, 2.151621161, 2.84872636, 0.829716540],
                          [0.721398745, 0.701933431, 2.571683647, 3.033219674, 1.240861292, 0.918585093, 2.300837562, 0.084058532, 3.02468093, 0.901128378],
                          [0.602291607, 0.272366672, 3.112129095, 2.251176550, 2.220469758, 2.503018530, 2.158829028, 2.906593698, 2.07943676, 2.152249278],
                          [0.508659922, 0.897082500, 0.534336638, 0.463481491, 0.415914667, 0.547196687, 2.993234168, 1.920102885, 0.74423709, 0.618257110],
                          [3.040408146, 2.422149474, 2.042527148, 2.119102513, 0.926953234, 2.544156645, 1.705974817, 2.173732966, 0.59878622, 2.729003869],
                          [3.124342488, 1.840771003, 1.783976727, 1.587334205, 1.307435032, 1.175014395, 0.230506585, 1.057695037, 1.28254244, 2.795696359],
                          [2.504012032, 2.232894954, 2.741109540, 1.686181937, 0.078694426, 0.781589316, 1.851317116, 3.120428890, 0.82601631, 0.436859660]])
        
    X_init = X[:n_init, :]
    y_init = test_func(X_init)
    return test_func, X_init, y_init


    
r"""
Candidate generation related functions
"""

INIT_OPTION_KEYS = {
    "alpha",
    "batch_limit",
    "init_batch_limit",
    "nonnegative",
    "gfn",
    "M",
    "L",
    "epsilon",
    "burnin",
    "n_gap",
    "hmc_type",
    "x0_type",
    "sample_around_best_sigma",
    "sample_around_best_subset_sigma",
    "sample_around_best_prob_perturb",
    "seed",
}



def gen_batch_initial(
        acq_function: AcquisitionFunction,
        bounds: Tensor,
        q: int,
        num_restarts: int,
        raw_samples: int,
        options: Optional[Dict[str, Union[bool, float, int]]] = None,
) -> Tensor:
    r""" Generate a batch of initial conditions for sampling

    Args: 
        acq_function: The acquisition function to be sampled
        bounds: A `2 x d` tensor of lower and upper bounds for each column
        q: The number of candidates to consider
        num_restarts: The number of starting points for multistart acquisition sampling
        raw_samples: The number of raw samples to consider in the initialization heuristic
        options: Options for initial condition generation. 

    Returns:
        A `num_restarts x q x d` tensor of initial conditions
    """

    options = options or {}
    seed: Optional[int] = options.get("seed")
    batch_limit: Optional[int] = options.get("init_batch_limit", options.get("batch_limit"))
    batch_initial_arms: Tensor
    device = bounds.device
    bounds_cpu = bounds.cpu()
    init_kwargs = {}
    x0_type = options.get("x0_type", "qmc")

    factor, max_factor = 1, 5
    
    q = 1 if q is None else q

    if options.get("nonnegative") or is_nonnegative(acq_function):
        init_func = initialize_q_batch_nonneg
        if "alpha" in options:
            init_kwargs["alpha"] = options.get("alpha")
    else:
        init_func = initialize_q_batch
            
    while factor < max_factor:
        n = raw_samples * factor
        if x0_type == "qmc":
            X_rnd = draw_sobol_samples(bounds = bounds_cpu, n = n, q = q)
        elif x0_type == "best":
            X_best_rnd = sample_points_around_best(
                acq_function = acq_function,
                n_discrete_points = n * q,
                sigma = options.get("sample_around_best_sigma", 1e-3),
                bounds = bounds,
                subset_sigma = options.get("sample_around_best_subset_sigma", 1e-1),
                prob_perturb = options.get("sample_around_best_prob_perturb"),
            )
            X_rnd = X_best_rnd.view(n, q, bounds.shape[-1]).cpu()
        elif x0_type == "lhs":
            dim = bounds.shape[-1]
            engine = LatinHypercube(d = dim)
            X_rnd = engine.random(n = n * q)
            X_rnd = torch.from_numpy(X_rnd).view(n, q, bounds.shape[-1]).to(torch.float32)
        elif x0_type == "unif":
            X_rnd_nlzd = torch.rand(
                n, q, bounds_cpu.shape[-1], dtype = bounds.dtype
            )
            X_rnd = bounds_cpu[0] + (bounds_cpu[1] - bounds_cpu[0]) * X_rnd_nlzd

        with torch.no_grad():
            if batch_limit is None:
                batch_limit = X_rnd.shape[0]
            Y_rnd_list = []
            start_idx = 0
            while start_idx < X_rnd.shape[0]:
                end_idx = min(start_idx + batch_limit, X_rnd.shape[0])
                Y_rnd_curr = acq_function(
                    X_rnd[start_idx:end_idx].to(device = device)
                )
                Y_rnd_list.append(Y_rnd_curr)
                start_idx += batch_limit
            Y_rnd = torch.cat(Y_rnd_list)
            batch_initial_conditions = init_func(
                X=X_rnd, Y=Y_rnd, n=num_restarts, **init_kwargs
            ).to(device=device)
            if factor < max_factor:
                factor += 1
                if seed is not None:
                    seed += 1
    return batch_initial_conditions
        
    


def simu_candidates(
        initial_conditions: Tensor,
        acquisition_function: AcquisitionFunction,
        bounds: Optional[Union[float, Tensor]] = None,
        options: Optional[Dict[str, Any]] = None,
) -> Tuple[Tensor, Tensor]:
    r""" Simulate a set of candidates using HMC / NUTS

    Args: 
        initial_conditions: Starting points for optimization.
        acquisition_function: Acquisition function to be used.
        lower_bounds: Minimum values for each column of initial_conditions.
        upper_bounds: Maximum values for each column of initial_conditions.
        options: Options used to control the sampling including "method"
            and "maxiter". Select method for `hamiltorch.HMC` or 
            `hamiltorch.HMC_NUTS` using the "method" key. 

    Returns: 
        2-element tuple containing
    
        - The set of generated candidates
        - The acquisition value for each t-batch

    Example: 
        >>> EI = ExpectedImprovement(model, best_f)
        >>> bounds = torch.tensor([[0., 0.], [1., 2.]])
        >>> Xinit = gen_batch_initial_conditions(
                EI, bounds, q = 3, num_restarts = 25, raw_samples = 500
            )
        >>> batch_candidates, batch_acq_values = simu_candidates(
                initial_conditions = Xinit, 
                acquisition_function = EI,
                lower_bounds = bounds[0],
                upper_bounds = bounds[1],
            )
    """
    options = options or {}
    lower_bounds = bounds[0]
    upper_bounds = bounds[1]
    clamped_candidates = columnwise_clamp(
        X = initial_conditions, lower = lower_bounds, upper = upper_bounds
    )
    shapeX = clamped_candidates.shape

    gfname = options.get("gfn", "exp")
    if gfname == "exp":
        def log_prob(x):
            return acquisition_function(x)
    else:
        def log_prob(x):
            return torch.log(acquisition_function(x))

    L = options.get("L", 50)
    M = options.get("M", 100)
    epsilon = options.get("epsilon", 1e-2)
    burnin = options.get("burnin", 50)
    n_gap = options.get("n_gap", 5)
    hmc_type = options.get("hmc_type", "hmc")

    if hmc_type == "hmc":
        candidates = torch.stack([torch.stack(HMC(log_prob, clamped_candidates[i, :, :], M, L, epsilon, burnin, n_gap)) for i in range(shapeX[0])])
    elif hmc_type == "hmc_wc":
        candidates = torch.stack([torch.stack(HMC_wc(log_prob, clamped_candidates[i, :, :], M, L, epsilon, burnin, n_gap, bounds)) for i in range(shapeX[0])])
    else:
        raise NotImplementedError(
            "Other sampling methods are not implemented. "
            "Select between hmc / hmc_wc for now."
        )

    shapeCandidates = candidates.shape
    candidates = candidates.view(shapeCandidates[0] * shapeCandidates[1], shapeCandidates[2], shapeCandidates[3])
    clamped_candidates = columnwise_clamp(
        X = candidates, lower = lower_bounds, upper = upper_bounds
    )
    # clamped_candidates = torch.stack([clamped_candidates[i, :, :] for i in range(shapeX[0])])

    with torch.no_grad():
        batch_acquisition = acquisition_function(clamped_candidates)

    return clamped_candidates, batch_acquisition
    
    

    

r"""
Acquisition sampling method
"""

def sample_acqf(
        acq_function: AcquisitionFunction,
        bounds: Tensor,
        q: int,
        num_restarts: int,
        raw_samples: Optional[int] = None,
        options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
        post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
        **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    r"""Generate a set of candidates via multi-start sampling

    Args: 
        acq_function: An AcquisitionFunction.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of samples for initialization. This is required
            if `batch_initial_conditions` is not specified.
        options: Options for candidate generation.
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e., according to `round-trip`
            transformations).
        batch_initial_conditions: A tensor to specify the initial conditions. Set
            this if you do not want to use default initialization strategy.
        return_best_only: If False, outputs the solutions corresponding to all
            random restart initializations of the optimization.
        kwargs: Additonal keyword arguments.

    Returns:
        A two-element tuple containing

        - a `(num_restarts) x q x d`-dim tensor of generated candidates. 
        - a tensor of associated acquisition values.

    Example:
        >>> # generate `q = 2` candidates jointly using 20 restarts
        >>> # and 512 raw samples
        >>> candidates, acq_value = sample_acqf(EI, bounds, 2, 20, 512)

    """

    if q > 1:
        candidate_list, acq_value_list = [], []
        base_X_pending = acq_function.X_pending

        for i in range(q):
            candidate, acq_value = sample_acqf(
                acq_function = acq_function,
                bounds = bounds,
                q = 1,
                num_restarts = num_restarts,
                raw_samples = raw_samples,
                options = options or {},
                post_processing_func = post_processing_func,
                batch_initial_conditions = None,
                return_best_only = True,
            )
            candidate_list.append(candidate)
            acq_value_list.append(acq_value)
            candidates = torch.cat(candidate_list, dim = -2)
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim = -2)
                if base_X_pending is not None
                else candidates
            )
            logger.info(f"Generated sequential candidate {i+1} of {q}")

        # Reset acq_func to previous X_pending state
        acq_function.set_X_pending(base_X_pending)
        return candidates, torch.stack(acq_value_list)

    options = options or {}
    batch_initial_conditions = options.get("batch_initial_conditions", None)

    if batch_initial_conditions is None:
        batch_initial_conditions = gen_batch_initial(
            acq_function = acq_function,
            bounds = bounds, 
            q = q, 
            num_restarts = num_restarts, 
            raw_samples = raw_samples, 
            options = options
        )
        
    batch_candidates_list: List[Tensor] = []
    batch_acq_values_list: List[Tensor] = []
    batch_limit: int = options.get("batch_limit", num_restarts)
    batched_ics = batch_initial_conditions.split(batch_limit)

    for i, batched_ics_ in enumerate(batched_ics):
        # sample via acqusiition function
        batch_candidates_curr, batch_acq_values_curr = simu_candidates(
            initial_conditions = batched_ics_,
            acquisition_function = acq_function,
            bounds = bounds,
            options = options,
        )
        batch_candidates_list.append(batch_candidates_curr)
        batch_acq_values_list.append(batch_acq_values_curr)
        logger.info(f"Generated candidate batch {i+1} of {len(batched_ics)}.")
    batch_candidates = torch.cat(batch_candidates_list)
    batch_acq_values = torch.cat(batch_acq_values_list)

    if post_processing_func is not None:
        n_sugs = options.get("n_sugs", 1)
        X_keep = options.get("X_keep")
        select_criteria = options.get("select_criteria")
        sel_func = options.get("sel_func")
        batch_candidates = post_processing_func(batch_candidates, X_keep, select_criteria, n_sugs, sel_func)

    batch_acq_values = acq_function(batch_candidates)
    # return_best_only = options.get("return_best_only", True)
    # if return_best_only:
    #     best = torch.argmax(batch_acq_values.view(-1), dim = 0)
    #     batch_candidates = batch_candidates.view(-1, len(bounds[0]))[best]
    #     batch_acq_values = batch_acq_values.view(-1)[best]

    return batch_candidates, batch_acq_values



def sample_acqf_list(
        acq_function_list: List[AcquisitionFunction],
        bounds: Tensor,
        num_restarts: int,
        raw_samples: Optional[int] = None,
        options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
        post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
        return_best_only: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""Generate a list of candidates from a list of acquisition functions

    Args: 
        acq_function_list: A list of acqusition functions
        bounds: A `2 x d` tesnor of lower and upper boudns for each column
        num_restarts: Number of starting points for multiple initial acquisition sampling
        raw_samples: Number of samples for initialization
        options: Options for candidates generation
        post_processing_func: A function that post-processes an result appropriately

    Returns:
        A two-element tuple containing
        - a `q x d`-dim tensor of generated candidates
        - a `q`- dim tensor of expected acqusiition values, where the value at index 
          `i` is the acquisition value conditional on having observed all candidates 
          except candidate `i`. 
    """
    if not acq_function_list:
        raise ValueError("acq_function_list must be non-empty")

    candidate_list, acq_value_list = [], []
    candidates = torch.tensor([], device = bounds.device, dtype = bounds.dtype)
    base_X_pending = acq_function_list[0].X_pending

    for acq_function in acq_function_list:
        if candidate_list:
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim = -2)
                if base_X_pending is not None
                else candidates
            )
        candidate, acq_value = sample_acqf(
            acq_function = acq_function,
            bounds = bounds,
            q = 1,
            num_restarts = num_restarts,
            raw_samples = raw_samples,
            options = options or {},
            post_processing_func = post_processing_func,
            return_best_only = return_best_only, 
        )
        candidate_list.append(candidate)
        acq_value_list.append(acq_value)
        candidates = torch.cat(candidate_list, dim = -2)
    return candidates, torch.stack(acq_value_list)

r""" set different
"""

def setdiff(a, b):
    res = []
    for aa in a:
        if aa not in b:
            res.append(aa)
    return res


r"""AugmentedLHS 
"""

def AugmentedLHS(Xs, m):
    r"""

    Args:
        Xs: Tensor, existing locations
        m: int, number of additional samples required
    """

    N = Xs.shape[0]
    K = Xs.shape[1]

    colvec = torch.sort(torch.rand(K)).indices
    rowvec = torch.sort(torch.rand(N + m)).indices

    B = torch.zeros(N + m, K)

    for j in colvec:
        newrow = 0
        for i in rowvec:
            flag = torch.logical_and(i / (N + m) <= Xs[:, j], Xs[:, j] <= (i + 1) / (N + m))
            if torch.any(~flag):
                newrow += 1
                B[newrow - 1, j] = (torch.rand(1) + (i - 1)) / (N + m)

    return B[:m, :]



r"""Post-processing function for selection
"""

def post_process_func(
        candidates,
        X_keep, 
        select_criteria,
        n_sugs,
        sel_func,
):
    r"""
    Args:
        candidates: A Tensor with suggested candidates
        X_keep: A Tensor with currently existing points
        select_criteria: Dict, with selection criteria
        n_sugs: int, number of suggested points
        sel_func: Callable, function to select points 
    """

    dim = X_keep.shape[-1]

    # check distance between suggestions and their distance between existing points
    if dim > 1:
        squeeze_cands = torch.unique(candidates.squeeze(), dim = 0)
    else:
        squeeze_cands = torch.unique(candidates.view(-1, 1), dim = 0)
        
    dist_bws = torch.triu(torch.cdist(squeeze_cands, squeeze_cands))
    dist_bws = dist_bws.fill_diagonal_(0.)
    bws_type = select_criteria.get("bws_type", "quantile")
    bws_param = select_criteria.get("bws_param")

    if bws_type == "quantile":
        threshold = torch.quantile(dist_bws[dist_bws > 0.], bws_param)
    elif bws_type == "value":
        threshold = torch.sqrt(dim) * bws_param

    while torch.any(dist_bws[dist_bws > 0] < threshold):
        min_dist0 = dist_bws[dist_bws > 0].min()
        where_idx_2 = torch.where(dist_bws == min_dist0)
        where_idx = [x.cpu().detach().item() for x in where_idx_2[0]]
        mean_row = []
        mean_idx_all = []
        while len(where_idx) > 0:
            sel_idx = where_idx[0]
            mean_idxi = [x.cpu().detach().item() for x in torch.where(dist_bws[sel_idx, :] == min_dist0)[0]]
            mean_idxi.append(sel_idx)
            mean_idx_all.extend(mean_idxi)
            mean_row.append(squeeze_cands[mean_idxi].mean(dim = 0))
            where_idx = setdiff(where_idx, mean_idxi)
        keep_idx = setdiff(torch.arange(squeeze_cands.shape[0]), mean_idx_all)
        squeeze_cands = torch.vstack([squeeze_cands[keep_idx, :], torch.vstack(mean_row)])
        dist_bws = torch.triu(torch.cdist(squeeze_cands, squeeze_cands))
        dist_bws = dist_bws.fill_diagonal_(0.)
        # print(squeeze_cands.shape[0])

    bwe_type = select_criteria.get("bwe_type", "quantile")
    bwe_param = select_criteria.get("bwe_param", 0.5)
    bwe_threshold = select_criteria.get("bwe_threshold")

    dist_bwe = torch.cdist(squeeze_cands, X_keep)

    if bwe_type == "quantile":
        dist_vec = torch.tensor([torch.quantile(dist_bwe[i, ], bwe_param) for i in range(squeeze_cands.shape[0])])
    elif bwe_type == "mean":
        dist_vec = dist_bwe.mean(dim = 1)

    keep_idx = dist_vec > bwe_threshold
    if sum(keep_idx).cpu().detach().item() > 0:
        squeeze_cands = squeeze_cands[keep_idx]

    # check with criteria
    if squeeze_cands.shape[0] > n_sugs:
        acqs = torch.tensor([sel_func(squeeze_cands[i].view(1, -1)) for i in range(squeeze_cands.shape[0])])
        acq_idx = torch.sort(acqs, descending = True).indices
        new_theta = squeeze_cands[acq_idx[:n_sugs]]
    elif squeeze_cands.shape[0] < n_sugs:
        n_add =	n_sugs - squeeze_cands.shape[0]
        X_add = AugmentedLHS(torch.vstack([X_keep, squeeze_cands]), n_add)
        new_theta = torch.vstack([squeeze_cands, X_add])
    else:
        new_theta = squeeze_cands

    return new_theta


r"""Determine beta in UCB
"""
def choose_beta(dim, Niter, delta = 0.1):
    res = np.power(Niter * np.pi, 2) / (6. * delta)
    return 2 * np.log(dim * res)



r"""Screen X and Y to get a subset
"""
def ScreenXY(X, Y, screen_criteria):
    select_type = screen_criteria.get("type")

    if select_type == "quantile":
        select_gamma = screen_criteria.get("gamma")
        idx = Y > torch.quantile(Y, select_gamma)
        idx = [x.item() for x in idx.cpu().detach()]
        if sum(idx) > 5:
            selY = Y[idx]
            selX = X[idx]
        else:
            selY = Y
            selX = X
    elif select_type == "sample":
        select_N = screen_criteria.get("N")
        select_gamma = screen_criteria.get("gamma")
        idx = Y > torch.quantile(Y, select_gamma)
        idx_in = [x.item() for x in idx.cpu().detach()]
        idx_nin = [not x.item() for x in idx.cpu().detach()]
        if sum(idx_in) > select_N:
            idx1 = torch.randint(sum(idx_in), (select_N, ))
            idx2 = torch.randint(sum(idx_nin), (select_N, ))
            X1 = X[idx_in]
            Y1 = Y[idx_in]
            X2 = X[idx_nin]
            Y2 = Y[idx_nin]
            X1 = X1[idx1]
            Y1 = Y1[idx1]
            X2 = X2[idx2]
            Y2 = Y2[idx2]
            selX = torch.vstack([X1, X2])
            selY = torch.vstack([Y1, Y2])
        elif sum(idx_in) < 5:
            selX = X
            selY = Y
        else:
            N = sum(idx_in)
            idx2 = torch.randint(sum(idx_nin), (N, ))
            X1 = X[idx_in]
            Y1 = Y[idx_in]
            X2 = X[idx_nin]
            Y2 = Y[idx_nin]
            X2 = X2[idx2]
            Y2 = Y2[idx2]
            selX = torch.vstack([X1, X2])
            selY = torch.vstack([Y1, Y2])
    return selX, selY


r"""Selection other than the current acquisition function
"""
def selection_func(x, model, sel_options):
    sel_type = sel_options.get("type")
    if sel_type == "ei":
        best_f = sel_options.get("best_f")
        acq = ExpectedImprovement(model, best_f = best_f)
        res = acq(x).cpu().detach().item()
    elif sel_type == "adj_ei":
        best_f = sel_options.get("best_f")
        X_keep = sel_options.get("X_keep")
        acq = AdjustedDistExpectedImprovement(model = model, best_f = best_f, X_keep = X_keep, maximize = True)
        res = acq(x).cpu().detach().item()
    elif sel_type == "ei_t":
        best_f = sel_options.get("threshold_value")
        acq = ExpectedImprovement(model = model, best_f = best_f, maximize = True)
        res = acq(x).cpu().detach().item()
    elif sel_type == "adj_ei_t":
        best_f = sel_options.get("threshold_value")
        X_keep = sel_options.get("X_keep")
        acq = AdjustedDistExpectedImprovement(model = model, best_f = best_f, X_keep = X_keep, maximize = True)
        res = acq(x).cpu().detach().item()
    elif sel_type == "poi":
        best_f = sel_options.get("best_f")
        acq = ProbabilityOfImprovement(model = model, best_f = best_f, maximize = True)
        res = acq(x).cpu().detach().item()
    elif sel_type == "poi_t":
        best_f = sel_options.get("threshold_value")
        acq = ProbabilityOfImprovement(model = model, best_f = best_f, maximize = True)
        res = acq(x).cpu().detach().item()
    elif sel_type == "adj_poi":
        best_f = sel_options.get("best_f")
        X_keep = sel_options.get("X_keep")
        acq = AdjustedDistProbabilityOfImprovement(model = model, best_f = best_f, X_keep = X_keep, maximize = True)
        res = acq(x).cpu().detach().item()
    elif sel_type == "adj_poi_t":
        best_f = sel_options.get("threshold_value")
        X_keep = sel_options.get("X_keep")
        acq = AdjustedDistProbabilityOfImprovement(model = model, best_f = best_f, X_keep = X_keep, maximize = True)
        res = acq(x).cpu().detach().item()
    elif sel_type == "ucb":
        dim = sel_options.get("dim")
        curr_iter = sel_options.get("curr_iter")
        beta = choose_beta(dim, curr_iter + 1)
        acq = UpperConfidenceBound(model = model, beta = beta, maximize = True)
        res = acq(x).cpu().detach().item()
    elif sel_type == "adj_ucb":
        dim = sel_options.get("dim")
        curr_iter = sel_options.get("curr_iter")
        X_keep = sel_options.get("X_keep")
        beta = choose_beta(dim, curr_iter + 1)
        acq = AdjustedDistUpperConfidenceBound(model = model, beta = beta, X_keep = X_keep, maximize = True)
        res = acq(x).cpu().detach().item()
    elif sel_type == "pr": 
        best_f = sel_options.get("best_f")
        acq = ProbabilityRatio(model = model, best_f = best_f, maximize = True)
        res = acq(x).cpu().detach().item()
    elif sel_type == "pr_t":
        best_f = sel_options.get("threshold_value")
        acq = ProbabilityRatio(model = model, best_f = best_f, maximize = True)
        res = acq(x).cpu().detach().item()
    elif sel_type == "adj_pr":
        best_f = sel_options.get("best_f")
        X_keep = sel_options.get("X_keep")
        acq = AdjustedDistProbabilityRatio(model = model, best_f = best_f, X_keep = X_keep, maximize = True)
        res = acq(x).cpu().detach().item()
    elif sel_type == "adj_pr_t":
        best_f = sel_options.get("threshold_value")
        X_keep = sel_options.get("X_keep")
        acq = AdjustedDistProbabilityRatio(model = model, best_f = best_f, X_keep = X_keep, maximize = True)
        res = acq(x).cpu().detach().item()
    elif sel_type == "mu":
        res = model.posterior(x).mean
    elif sel_type == "grad_mu":
        x.requires_grad = True
        mu = model.posterior(x).mean.squeeze()
        mu.backward()
        res = 1. / torch.mean(torch.abs(x.grad))
    return res



