"""
-------------------------------------------------------------------------------
The Autonomous Capabilities Team (ACT3) Deep Reinforcement Learning (D-RL) Environment

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
-------------------------------------------------------------------------------
"""
import abc
import random
from functools import partial

from pydantic import BaseModel
from ray import tune


class ParametersPPO:
    """Utility functions for processing hparam searches in the  framework for PPO algorithm
    https://github.com/ray-project/ray/blob/00922817b66ee14ba215972a98f416f3d6fef1ba/rllib/agents/ppo/ppo.py
    https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
    https://blog.openai.com/openai-five/
    https://docs.ray.io/en/master/tune/api_docs/trainable.html#advanced-reusing-actors
    """

    LAMBDA_MIN = 0.9
    LAMBDA_MAX = 1.0
    LAMBDA_DIST = tune.uniform(LAMBDA_MIN, LAMBDA_MAX)

    @staticmethod
    def LAMBDA_RANGE(spec):
        """Sets the default search space for HPARAM
        GAE Parameter Lambda Range: 0.9 to 1
        GAE Parameter Lambda also known as: GAE Parameter (lambda) (PPO Paper), lambda (RLlib),
        lambda (ppo2 baselines), lambda (ppo baselines), lambda (Unity ML), gae_lambda (TensorForce)
        """
        return ParametersPPO.LAMBDA_DIST

    VF_LOSS_COEFF_MIN = 0.5
    VF_LOSS_COEFF_MAX = 1.0
    VF_LOSS_COEFF_DIST = tune.uniform(VF_LOSS_COEFF_MIN, VF_LOSS_COEFF_MAX)

    @staticmethod
    def VF_LOSS_COEFF_RANGE(spec):
        """Sets the default search space for HPARAM
        Value Function Coefficient Range: 0.5, 1
        Value Function Coefficient also known as: VF coeff. (PPO Paper), vf_loss_coef (RLlib),
        vf_coef (ppo2 baselines), (ppo baselines: unclear), (Unity ML: unclear), (TensorForce: unclear)
        """
        return ParametersPPO.VF_LOSS_COEFF_DIST

    ENTROPY_COEFF_MIN = 0.00
    ENTROPY_COEFF_MAX = 0.01
    ENTROPY_COEFF_DIST = tune.uniform(ENTROPY_COEFF_MIN, ENTROPY_COEFF_MAX)

    @staticmethod
    def ENTROPY_COEFF_RANGE(spec):
        """Sets the default search space for HPARAM
        Entropy Coefficient Range: 0 to 0.01
        Entropy Coefficient also known as: Entropy coeff. (PPO Paper), entropy_coeff (RLlib),
        ent_coeff (ppo2 baselines), entcoeff (ppo baselines), beta (Unity ML), entropy_regularization (TensorForce)
        """
        return ParametersPPO.ENTROPY_COEFF_DIST

    CLIP_PARAM_MIN = 0.1
    CLIP_PARAM_MAX = 0.3
    CLIP_PARAM_DIST = tune.choice([0.1, 0.2, 0.3])

    @staticmethod
    def CLIP_PARAM_RANGE(spec):
        """Sets the default search space for HPARAM
        Clipping Range: 0.1, 0.2, 0.3
        Clipping also known as: Clipping parameter epsilon (PPO Paper), clip_param (RLlib),
        cliprange (ppo2 baselines), clip_param (ppo baselines), epsilon (Unity ML),
        likelihood_ratio_clipping (TensorForce)
        """
        return ParametersPPO.CLIP_PARAM_DIST

    KL_TARGET_MIN = 0.003
    KL_TARGET_MAX = 0.03
    KL_TARGET_DIST = tune.uniform(KL_TARGET_MIN, KL_TARGET_MAX)

    @staticmethod
    def KL_TARGET_RANGE(spec):
        """Sets the default search space for HPARAM
        https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
        The KL penalty implementation (third line in the above picture) is available in RLlib's PPO
        implementation. The parameters kl_coeff (initial coefficient for KL divergence) and kl_target
        can be used for the KL implementation.
        KL Target Range: 0.003 to 0.03
        KL Initialization Range: 0.3 to 1  --- KL_COEFF IN RLLIB
        """
        return ParametersPPO.KL_COEFF_DIST

    KL_COEFF_MIN = 0.2  # RLLIB Default
    KL_COEFF_MAX = 1.0  # https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
    KL_COEFF_DIST = tune.uniform(KL_COEFF_MIN, KL_COEFF_MAX)

    @staticmethod
    def KL_COEFF_RANGE(spec):
        """Sets the default search space for HPARAM
        The KL penalty implementation (third line in the above picture) is available in RLlib's PPO
        implementation. The parameters kl_coeff (initial coefficient for KL divergence) and kl_target
        can be used for the KL implementation.
        KL Target Range: 0.003 to 0.03
        KL Initialization Range: 0.3 to 1  --- KL_COEFF IN RLLIB
        """
        return ParametersPPO.KL_COEFF_DIST

    GAMMA_MIN = 0.8000
    GAMMA_MAX = 0.9997
    GAMMA_DIST = tune.uniform(GAMMA_MIN, GAMMA_MAX)

    @staticmethod
    def GAMMA_RANGE(spec):
        """Sets the default search space for HPARAM
        Discount Factor Gamma Range: 0.99 (most common), 0.8 to 0.9997
        Discount Factor Gamma also known as: Discount (gamma) (PPO Paper), gamma (RLlib), gamma (ppo2 baselines),
        gamma (ppo baselines), gamma (Unity ML), discount (TensorForce)
        """
        return ParametersPPO.GAMMA_DIST

    LR_MIN = 5e-6
    LR_MAX = 0.003
    LR_DIST = tune.uniform(LR_MIN, LR_MAX)

    @staticmethod
    def LR_RANGE(spec):
        """Sets the default search space for HPARAM
        Learning Rate Range: 0.003 to 5e-6
        Learning Rate also known as: Adam stepsize (PPO Paper), sgd_stepsize (RLlib), lr (ppo2 baselines),
        (ppo baselines: unclear), learning_rate (Unity ML), learning_rate (TensorForce)
        """
        return ParametersPPO.LR_DIST

    NSGD_MIN = 3
    NSGD_MAX = 30
    NSGD_DIST = tune.choice(list(range(NSGD_MIN, NSGD_MAX + 1)))

    @staticmethod
    def NSGD_RANGE(spec):
        """Sets the default search space for HPARAM
        https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
        Epoch Range: 3 to 30
        Epoch also known as: Num. epochs (PPO paper), num_sgd_iter (RLlib), epochs (ppo2 baselines),
        optim_epochs (ppo baselines), num_epoch (Unity ML), (TensorForce: unclear)
        """
        return ParametersPPO.NSGD_DIST

    SGD_MINIBATCH_SIZE_MIN = 128
    SGD_MINIBATCH_SIZE_MAX = 4096
    SGD_MINIBATCH_SIZE_DIST = tune.choice([128, 256, 512, 1024, 2048, 4096])

    @staticmethod
    def SGD_MINIBATCH_SIZE_RANGE(spec):
        """Sets the default search space for HPARAM"""
        return ParametersPPO.SGD_MINIBATCH_SIZE_DIST

    TRAIN_BATCH_SIZE_MIN = 4096
    TRAIN_BATCH_SIZE_MAX = 160000
    TRAIN_BATCH_SIZE_INC = 256

    # [4096, 4352, 4608, 4864, 5120, 5376, 5632, 5888, 6144, 6400, 6656, 6912, 7168, 7424, 7680, 7936, 8192, 8448,
    #  8704, 8960, 9216, 9472, 9728, 9984, 10240, 10496, 10752, 11008, 11264, 11520, 11776, 12032, 12288, 12544,
    #  12800, 13056, 13312, 13568, 13824, 14080, 14336, 14592, 14848, 15104, 15360, 15616, 15872, 16128, 16384,
    #  16640, 16896, 17152, 17408, 17664, 17920, 18176, 18432, 18688, 18944, 19200, 19456, 19712, 19968, 20224,
    #  20480, 20736, 20992, 21248, 21504, 21760, 22016, 22272, 22528, 22784, 23040, 23296, 23552, 23808, 24064,
    #  24320, 24576, 24832, 25088, 25344, 25600, 25856, 26112, 26368, 26624, 26880, 27136, 27392, 27648, 27904,
    #  28160, 28416, 28672, 28928, 29184, 29440, 29696, 29952, 30208, 30464, 30720, 30976, 31232, 31488, 31744,
    #  32000, 32256, 32512, 32768, 33024, 33280, 33536, 33792, 34048, 34304, 34560, 34816, 35072, 35328, 35584,
    #  35840, 36096, 36352, 36608, 36864, 37120, 37376, 37632, 37888, 38144, 38400, 38656, 38912, 39168, 39424,
    #  39680, 39936, 40192, 40448, 40704, 40960, 41216, 41472, 41728, 41984, 42240, 42496, 42752, 43008, 43264,
    #  43520, 43776, 44032, 44288, 44544, 44800, 45056, 45312, 45568, 45824, 46080, 46336, 46592, 46848, 47104,
    #  47360, 47616, 47872, 48128, 48384, 48640, 48896, 49152, 49408, 49664, 49920, 50176, 50432, 50688, 50944,
    #  51200, 51456, 51712, 51968, 52224, 52480, 52736, 52992, 53248, 53504, 53760, 54016, 54272, 54528, 54784,
    #  55040, 55296, 55552, 55808, 56064, 56320, 56576, 56832, 57088, 57344, 57600, 57856, 58112, 58368, 58624,
    #  58880, 59136, 59392, 59648, 59904, 60160, 60416, 60672, 60928, 61184, 61440, 61696, 61952, 62208, 62464,
    #  62720, 62976, 63232, 63488, 63744, 64000, 64256, 64512, 64768, 65024, 65280, 65536, 65792, 66048, 66304,
    #  66560, 66816, 67072, 67328, 67584, 67840, 68096, 68352, 68608, 68864, 69120, 69376, 69632, 69888, 70144,
    #  70400, 70656, 70912, 71168, 71424, 71680, 71936, 72192, 72448, 72704, 72960, 73216, 73472, 73728, 73984,
    #  74240, 74496, 74752, 75008, 75264, 75520, 75776, 76032, 76288, 76544, 76800, 77056, 77312, 77568, 77824,
    #  78080, 78336, 78592, 78848, 79104, 79360, 79616, 79872, 80128, 80384, 80640, 80896, 81152, 81408, 81664,
    #  81920, 82176, 82432, 82688, 82944, 83200, 83456, 83712, 83968, 84224, 84480, 84736, 84992, 85248, 85504,
    #  85760, 86016, 86272, 86528, 86784, 87040, 87296, 87552, 87808, 88064, 88320, 88576, 88832, 89088, 89344,
    #  89600, 89856, 90112, 90368, 90624, 90880, 91136, 91392, 91648, 91904, 92160, 92416, 92672, 92928, 93184,
    #  93440, 93696, 93952, 94208, 94464, 94720, 94976, 95232, 95488, 95744, 96000, 96256, 96512, 96768, 97024,
    #  97280, 97536, 97792, 98048, 98304, 98560, 98816, 99072, 99328, 99584, 99840, 100096, 100352, 100608, 100864,
    #  101120, 101376, 101632, 101888, 102144, 102400, 102656, 102912, 103168, 103424, 103680, 103936, 104192,
    #  104448, 104704, 104960, 105216, 105472, 105728, 105984, 106240, 106496, 106752, 107008, 107264, 107520,
    #  107776, 108032, 108288, 108544, 108800, 109056, 109312, 109568, 109824, 110080, 110336, 110592, 110848,
    #  111104, 111360, 111616, 111872, 112128, 112384, 112640, 112896, 113152, 113408, 113664, 113920, 114176,
    #  114432, 114688, 114944, 115200, 115456, 115712, 115968, 116224, 116480, 116736, 116992, 117248, 117504,
    #  117760, 118016, 118272, 118528, 118784, 119040, 119296, 119552, 119808, 120064, 120320, 120576, 120832,
    #  121088, 121344, 121600, 121856, 122112, 122368, 122624, 122880, 123136, 123392, 123648, 123904, 124160,
    #  124416, 124672, 124928, 125184, 125440, 125696, 125952, 126208, 126464, 126720, 126976, 127232, 127488,
    #  127744, 128000, 128256, 128512, 128768, 129024, 129280, 129536, 129792, 130048, 130304, 130560, 130816,
    #  131072, 131328, 131584, 131840, 132096, 132352, 132608, 132864, 133120, 133376, 133632, 133888, 134144,
    #  134400, 134656, 134912, 135168, 135424, 135680, 135936, 136192, 136448, 136704, 136960, 137216, 137472,
    #  137728, 137984, 138240, 138496, 138752, 139008, 139264, 139520, 139776, 140032, 140288, 140544, 140800,
    #  141056, 141312, 141568, 141824, 142080, 142336, 142592, 142848, 143104, 143360, 143616, 143872, 144128,
    #  144384, 144640, 144896, 145152, 145408, 145664, 145920, 146176, 146432, 146688, 146944, 147200, 147456,
    #  147712, 147968, 148224, 148480, 148736, 148992, 149248, 149504, 149760, 150016, 150272, 150528, 150784,
    #  151040, 151296, 151552, 151808, 152064, 152320, 152576, 152832, 153088, 153344, 153600, 153856, 154112,
    #  154368, 154624, 154880, 155136, 155392, 155648, 155904, 156160, 156416, 156672, 156928, 157184, 157440,
    #  157696, 157952, 158208, 158464, 158720, 158976, 159232, 159488, 159744, 160000]
    TRAIN_BATCH_SIZE_DIST = tune.choice(list(range(2**12, 160000 + TRAIN_BATCH_SIZE_INC, TRAIN_BATCH_SIZE_INC)))

    @staticmethod
    def TRAIN_BATCH_SIZE_RANGE(spec):
        """Sets the default search space for HPARAM"""
        return ParametersPPO.TRAIN_BATCH_SIZE_DIST

    @staticmethod
    def ppo_hyperparameters() -> dict:
        """PPO hyper parameters for hparam search
        https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe

        Returns:
            dict -- model configuration
        """
        ppo_hparams = {}
        ppo_hparams["lambda"] = tune.sample_from(ParametersPPO.LAMBDA_RANGE)
        ppo_hparams["vf_loss_coeff"] = tune.sample_from(ParametersPPO.VF_LOSS_COEFF_RANGE)
        ppo_hparams["entropy_coeff"] = tune.sample_from(ParametersPPO.ENTROPY_COEFF_RANGE)
        ppo_hparams["clip_param"] = tune.sample_from(ParametersPPO.CLIP_PARAM_RANGE)
        ppo_hparams["gamma"] = tune.sample_from(ParametersPPO.GAMMA_RANGE)
        ppo_hparams["lr"] = tune.sample_from(ParametersPPO.LR_RANGE)
        ppo_hparams["num_sgd_iter"] = tune.sample_from(ParametersPPO.NSGD_RANGE)
        ppo_hparams["sgd_minibatch_size"] = tune.sample_from(ParametersPPO.SGD_MINIBATCH_SIZE_RANGE)
        ppo_hparams["train_batch_size"] = tune.sample_from(ParametersPPO.TRAIN_BATCH_SIZE_RANGE)
        ppo_hparams["kl_coeff"] = tune.sample_from(ParametersPPO.KL_COEFF_RANGE)
        ppo_hparams["kl_target"] = tune.sample_from(ParametersPPO.KL_TARGET_RANGE)
        return ppo_hparams

    @staticmethod
    def sample_ppo_hyperparameters() -> dict:
        """PPO hyper parameters for hparam search
        https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe

        Returns:
            dict -- model configuration
        """
        ppo_hparams = ParametersPPO.ppo_hyperparameters()
        for k, v in ppo_hparams.items():
            ppo_hparams[k] = v.sample()
        return ppo_hparams

    @staticmethod
    def pbt_ppo_explore(config: dict) -> dict:
        """The following function links to the companion function above. Sets the clipping needed by PBT
        https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
        Arguments:
            config {dict} -- input config

        Returns:
            dict -- clipped config
        """

        def clip_parameter(config, parameter, parameter_max, parameter_min):
            if config[parameter] > parameter_max:
                config[parameter] = parameter_max
            elif config[parameter] < parameter_min:
                config[parameter] = parameter_min

        clip_parameter(config, "lambda", ParametersPPO.LAMBDA_MAX, ParametersPPO.LAMBDA_MIN)
        clip_parameter(config, "vf_loss_coeff", ParametersPPO.VF_LOSS_COEFF_MAX, ParametersPPO.VF_LOSS_COEFF_MIN)
        clip_parameter(config, "entropy_coeff", ParametersPPO.ENTROPY_COEFF_MAX, ParametersPPO.ENTROPY_COEFF_MIN)
        clip_parameter(config, "gamma", ParametersPPO.GAMMA_MAX, ParametersPPO.GAMMA_MIN)
        clip_parameter(config, "clip_param", ParametersPPO.CLIP_PARAM_MAX, ParametersPPO.CLIP_PARAM_MIN)
        clip_parameter(config, "lr", ParametersPPO.LR_MIN, ParametersPPO.LR_MAX)
        clip_parameter(config, "kl_coeff", ParametersPPO.KL_COEFF_MIN, ParametersPPO.KL_COEFF_MAX)
        clip_parameter(config, "kl_target", ParametersPPO.KL_TARGET_MIN, ParametersPPO.KL_TARGET_MAX)

        sgd_minibatch_size_str = "sgd_minibatch_size"
        train_batch_size_str = "train_batch_size"
        num_sgd_iter_str = "num_sgd_iter"
        clip_parameter(config, num_sgd_iter_str, ParametersPPO.NSGD_MAX, ParametersPPO.NSGD_MIN)
        config[num_sgd_iter_str] = int(config[num_sgd_iter_str])
        clip_parameter(config, sgd_minibatch_size_str, ParametersPPO.SGD_MINIBATCH_SIZE_MAX, ParametersPPO.SGD_MINIBATCH_SIZE_MIN)
        config[sgd_minibatch_size_str] = int(config[sgd_minibatch_size_str])
        clip_parameter(config, train_batch_size_str, ParametersPPO.TRAIN_BATCH_SIZE_MAX, ParametersPPO.TRAIN_BATCH_SIZE_MIN)
        if config[train_batch_size_str] < config[sgd_minibatch_size_str] * 2:
            config[train_batch_size_str] = config[sgd_minibatch_size_str] * 2
        config[train_batch_size_str] = int(config[train_batch_size_str])
        return config


class ParametersModel:
    """Holds the model parameters"""

    FC_LAYER_CHOICES = [32, 64, 128, 256, 512]
    FC_LAYER_COUNT = [2, 3, 4, 5, 6]

    @staticmethod
    def __get_layers(layer_count, FC_FILTER_LOWER_VALUES_THRESHOLD, layer_choices, MIN_LAYER_INDEX) -> list:
        model_layers: list = []
        for _ in range(layer_count):
            temp_layers = layer_choices[MIN_LAYER_INDEX:] if len(model_layers) < FC_FILTER_LOWER_VALUES_THRESHOLD else layer_choices
            if model_layers:
                model_layers.append(random.choice([x for x in temp_layers if x <= model_layers[-1]]))  # noqa: S311
            else:
                model_layers.append(random.choice(temp_layers))  # noqa: S311
        return model_layers

    @staticmethod
    def select_lstm_model() -> dict:
        """[summary]

        Returns
        -------
        dict
            [description]
        """
        model_config = ParametersModel.select_fully_connected_model()
        model_config["use_lstm"] = True
        model_config["max_seq_len"] = random.choice([2, 3, 5, 10])  # noqa: S311
        model_config["lstm_cell_size"] = random.choice([64, 128, 256, 512, 1024, 2048])  # noqa: S311
        model_config["vf_share_layers"] = True
        # model_config["vf_share_layers"] = random.choice([True, False])
        # model_config["lstm_use_prev_action"] = random.choice([True, False])
        # model_config["lstm_use_prev_reward"] = random.choice([True, False])
        # model_config["_time_major"] = random.choice([True, False])

        return model_config

    @staticmethod
    def select_fully_connected_model() -> dict:
        """[summary]

        Returns
        -------
        dict
            [description]
        """
        layer_count = random.choice(ParametersModel.FC_LAYER_COUNT)  # noqa: S311
        layer_choices = ParametersModel.FC_LAYER_CHOICES
        FC_FILTER_LOWER_VALUES_THRESHOLD = 2
        MIN_LAYER_INDEX = 2
        model_layers = ParametersModel.__get_layers(layer_count, FC_FILTER_LOWER_VALUES_THRESHOLD, layer_choices, MIN_LAYER_INDEX)

        model_config: dict = {}
        model_config["fcnet_hiddens"] = model_layers
        model_config["fcnet_activation"] = random.choice(["relu", "tanh"])  # noqa: S311

        return model_config

    @staticmethod
    def select_framestacking_model() -> dict:
        """[summary]

        Returns:
            dict -- [description]
        """
        model_config = ParametersModel.select_fully_connected_model()
        model_config["custom_model"] = "TorchFrameStack"
        model_config["custom_model_config"] = {}
        model_config["custom_model_config"]["num_frames"] = random.choice(list(range(1, 11)))  # noqa: S311
        model_config["custom_model_config"]["include_actions"] = random.choice([True, False])  # noqa: S311
        model_config["custom_model_config"]["include_rewards"] = random.choice([True, False])  # noqa: S311
        layer_count = random.choice(ParametersModel.FC_LAYER_COUNT)  # noqa: S311
        layer_choices = ParametersModel.FC_LAYER_CHOICES
        FC_FILTER_LOWER_VALUES_THRESHOLD = 2
        MIN_LAYER_INDEX = 2
        model_config["custom_model_config"]["post_fcnet_hiddens"] = ParametersModel.__get_layers(
            layer_count, FC_FILTER_LOWER_VALUES_THRESHOLD, layer_choices, MIN_LAYER_INDEX
        )

        return model_config

    @staticmethod
    def select_gtrxl_model() -> dict:
        """[summary]

        Returns
        -------
        dict
            [description]
        """
        model_config = ParametersModel.select_fully_connected_model()
        model_config["use_attention"] = False
        model_config["attention_num_transformer_units"] = random.choice(list(range(1, 6)))  # noqa: S311
        model_config["attention_dim"] = random.choice([64, 128, 256, 512, 1024, 2048])  # noqa: S311
        model_config["attention_num_heads"] = random.choice(list(range(1, 6)))  # noqa: S311
        model_config["attention_head_dim"] = random.choice([64, 128, 256, 512, 1024, 2048])  # noqa: S311
        model_config["attention_memory_inference"] = 50
        model_config["attention_memory_training"] = 50
        model_config["attention_position_wise_mlp_dim"] = random.choice([64, 128, 256, 512, 1024, 2048])  # noqa: S311
        # model_config["attention_init_gru_gate_bias"] = 2.0
        model_config["attention_use_n_prev_actions"] = random.choice(list(range(1, 11)))  # noqa: S311
        model_config["attention_use_n_prev_rewards"] = random.choice(list(range(1, 11)))  # noqa: S311
        return model_config

    @staticmethod
    def select_model(model_choices) -> dict:
        """The following function provides the start to exploring model configurations."""
        model_config_func = random.choice(model_choices)  # noqa: S311

        return model_config_func()


class BaseHparamSearchValidator(BaseModel):
    """
    Base Validator to subclass for Experiments subclassing BaseExperiment
    """


class BaseHparamSearch(abc.ABC):
    """
    Experiment provides an anstract class to run specific types of experiments
    this allows users to do specific setup steps or to run some sort of custom training
    loop
    """

    def __init__(self, **kwargs) -> None:
        self.config: BaseHparamSearchValidator = self.get_validator()(**kwargs)

    @staticmethod
    def get_validator() -> type[BaseHparamSearchValidator]:
        """Gets the validator

        Returns
        -------
        typing.Type[BaseHparamSearchValidator]
            The validator class
        """
        return BaseHparamSearchValidator

    @abc.abstractmethod
    def add_algorithm_hparams(self, rllib_config: dict, tune_config: dict) -> None:
        """Updates the configurations based on search alg

        Parameters
        ----------
        rllib_config : dict
            The rllib config
        tune_config : dict
            The tune config
        """


class HparamSearchValidator_Shared(BaseHparamSearchValidator):
    """
    Base Validator to subclass for search subclassing
    """

    # "The training result objective value attribute. Stopping procedures will use this attribute."
    metric: str = "episode_reward_mean"
    # One of {min, max}. Determines whether objective is minimizing or maximizing the metric attribute.
    mode: str = "max"
    # A training result attr to use for comparing time. Note that you can pass in something
    # non-temporal such as training_iteration as a measure of progress, the only requirement is
    # that the attribute should increase monotonically.
    time_attr: str = "timesteps_total"
    # The number of samples to collect during HPARAM search (trials)
    samples: int = 4


class HparamSearchValidator_PBT(HparamSearchValidator_Shared):
    """
    Base Validator to subclass for search subclassing
    """

    # The probability of resampling from the original distribution when applying hyperparam_mutations.
    # If not resampled, the value will be perturbed by a factor of 1.2 or 0.8 if continuous, or changed
    # to an adjacent value if discrete. Note that resample_probability by default is 0.25, thus
    # hyperparameter with a distribution may go out of the specific range.
    resample_probability: float = 0.25
    # (float) - Models will be considered for perturbation at this interval of time_attr. Note that
    # perturbation incurs checkpoint overhead, so you shouldn't set this to be too frequent.
    perturbation_interval: float = 4
    # (float) - Models will not be considered for perturbation before this interval of time_attr has
    # passed. This guarantees that models are trained for at least a certain amount of time or timesteps
    # before being perturbed.
    burn_in_period: float = 10


class HparamSearchPPO_PBT(BaseHparamSearch):
    """PPO PBT Search Space
    https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
    """

    @staticmethod
    def get_validator() -> type[HparamSearchValidator_PBT]:
        """gets the configuration for AHBS

        Returns
        -------
        typing.Type[HparamSearchValidator_AHBS]
            validator
        """
        return HparamSearchValidator_PBT

    def add_algorithm_hparams(self, rllib_config, tune_config) -> None:
        """Adds population based training to the configuration (TBD items to be added - default never add)

        Parameters
        ----------
        rllib_config : dict
            The experiment configuration
        tune_config : dict
            The tune configuration
        """

        # Postprocess the perturbed config to ensure it's still valid
        pbt = tune.schedulers.PopulationBasedTraining(
            time_attr=self.config.time_attr,  # type: ignore
            metric=self.config.metric,  # type: ignore
            mode=self.config.mode,  # type: ignore
            perturbation_interval=self.config.perturbation_interval,  # type: ignore
            resample_probability=self.config.resample_probability,  # type: ignore
            burn_in_period=self.config.burn_in_period,  # type: ignore
            log_config=True,
            # Specifies the mutations of these hyper params
            hyperparam_mutations={
                "lambda": ParametersPPO.LAMBDA_DIST,
                "clip_param": ParametersPPO.CLIP_PARAM_DIST,
                "lr": ParametersPPO.LR_DIST,
                "num_sgd_iter": ParametersPPO.NSGD_DIST,
                "sgd_minibatch_size": ParametersPPO.SGD_MINIBATCH_SIZE_DIST,
                "train_batch_size": ParametersPPO.TRAIN_BATCH_SIZE_DIST,
                "vf_loss_coeff": ParametersPPO.VF_LOSS_COEFF_DIST,
                "entropy_coeff": ParametersPPO.ENTROPY_COEFF_DIST,
                "gamma": ParametersPPO.GAMMA_DIST,
                "kl_coeff": ParametersPPO.KL_COEFF_DIST,
                "kl_target": ParametersPPO.KL_TARGET_DIST,
            },
            custom_explore_fn=ParametersPPO.pbt_ppo_explore,
        )
        # These params start off randomly drawn from a set.
        tune_config["scheduler"] = pbt
        tune_config["num_samples"] = self.config.samples  # type: ignore
        rllib_config.update(ParametersPPO.ppo_hyperparameters())


class HparamSearchValidator_AHBS(HparamSearchValidator_Shared):
    """
    Base Validator to subclass for search subclassing
    """

    # max time units per trial. Trials will be stopped after max_t time units (determined
    # by time_attr) have passed.
    max_t: float = 1e7
    # Brackets
    brackets: float = 1
    # Only stop trials at least this old in time. The units are the same as the attribute
    # named by time-attr.
    grace_period: float = 5e6

    include_lstm_search: bool = False
    inclue_fully_connected_search: bool = True
    include_frame_stacking_search: bool = False
    include_gtrxl_search: bool = False


class HparamSearchPPO_AHBS(BaseHparamSearch):
    """Asynchronous Hyper Band Example
    https://docs.ray.io/en/master/tune/examples/includes/async_hyperband_example.html
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs=kwargs)

        self._model_choices = []

        if self.config.include_lstm_search:  # type: ignore
            self._model_choices.append(ParametersModel.select_lstm_model)

        if self.config.inclue_fully_connected_search:  # type: ignore
            self._model_choices.append(ParametersModel.select_fully_connected_model)

        if self.config.include_frame_stacking_search:  # type: ignore
            self._model_choices.append(ParametersModel.select_framestacking_model)

        if self.config.include_gtrxl_search:  # type: ignore
            self._model_choices.append(ParametersModel.select_gtrxl_model)

    @staticmethod
    def get_validator() -> type[HparamSearchValidator_AHBS]:
        """gets the configuration for AHBS

        Returns
        -------
        typing.Type[HparamSearchValidator_AHBS]
            validator
        """
        return HparamSearchValidator_AHBS

    def add_algorithm_hparams(self, rllib_config, tune_config) -> None:
        """[summary]

        Parameters
        ----------
        run_or_experiment_config : [type]
            [description]
        tune_config : [type]
            [description]
        args : [type]
            [description]
        """
        ahbs = tune.schedulers.AsyncHyperBandScheduler(
            time_attr=self.config.time_attr,  # type: ignore
            metric=self.config.metric,  # type: ignore
            mode=self.config.mode,  # type: ignore
            max_t=self.config.max_t,  # type: ignore
            grace_period=self.config.grace_period,  # type: ignore
            brackets=self.config.brackets,  # type: ignore
        )
        tune_config["num_samples"] = self.config.samples  # type: ignore
        tune_config["scheduler"] = ahbs
        rllib_config.update(ParametersPPO.ppo_hyperparameters())
        rllib_config["model"] = tune.sample_from(partial(ParametersModel.select_model, self._model_choices))
