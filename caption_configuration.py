class GeneralConfig(object):
    def __init__(self):
        # Learning Rates
        self.lr_backbone = 1e-5
        self.lr = 1e-4

        # Epochs
        self.epochs = 30
        self.lr_drop = 20
        self.start_epoch = 0
        self.weight_decay = 1e-4

        # Backbone
        self.backbone = 'resnet50'
        self.position_embedding = 'sine'
        self.dilation = True

        # Basic
        self.device = 'cuda'
        self.seed = 42
        self.batch_size = 16
        self.num_workers = 8
        self.checkpoint = 'xxx.pth'
        self.save_path = './train_log/general/'
        self.clip_max_norm = 0.1

        # Transformer
        self.hidden_dim = 256
        self.pad_token_id = 0
        self.max_position_embeddings = 128
        self.layer_norm_eps = 1e-12
        self.dropout = 0.1
        self.vocab_size = 30522

        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 2048
        self.nheads = 8
        self.pre_norm = True

        # Dataset
        self.dir = 'Report'
        self.limit = 10000


class SwinConfig(object):
    def __init__(self):

        self.trans_use_classify = False
        # Learning Rates
        self.lr_backbone = 1e-5
        self.lr = 1e-4

        # Epochs
        self.epochs = 10
        self.lr_drop = 20
        self.start_epoch = 0
        self.weight_decay = 1e-4

        # Backbone
        self.backbone = 'swin'
        self.position_embedding = 'sine'
        self.dilation = True

        # Basic
        self.device = 'cuda'
        self.seed = 42
        self.batch_size = 8
        self.num_workers = 4
        self.checkpoint = ''
        self.clip_max_norm = 0.1

        # Transformer encoder
        self.swin_embed_dim = 96
        self.img_size = 224
        self.swin_depth = [2,2,2,2]
        self.swin_num_heads = [ 3, 6, 12, 24 ]
        self.swin_window_size = 7
        self.swin_resume = 'model_swin_6_0.5815_swin_3_val2.pth'

        # Transformer decoder
        self.hidden_dim = 256
        self.pad_token_id = 0
        self.max_position_embeddings = 128
        self.layer_norm_eps = 1e-12
        self.dropout = 0.1
        self.vocab_size = 30522

        self.dec_layers = 3
        self.enc_layers = 'swin'
        self.dim_feedforward = 2048
        self.nheads = 8
        self.pre_norm = True

        # Dataset
        self.dir = 'Report'
        self.limit = -1
        self.val_file = 'val.txt'
        self.test_file = 'test.txt'


class ClassifyConfig(object):
    def __init__(self):
        # Learning Rate
        self.lr_backbone = 1e-4
        self.lr = 1e-3

        # Epochs
        self.epochs = 15
        self.lr_drop = 20
        self.start_epoch = 0
        self.weight_decay = 5e-6

        # Basic
        self.device = 'cuda:1'
        self.seed = 42
        self.batch_size = 16
        self.num_workers = 8
        self.checkpoint = 'xxx.pth'
        self.saving_path = './train_log_stomach/classify/'
        self.clip_max_norm = 0.1
        self.num_class = 3

        # Dataset
        self.dir = 'Report'
        self.limit = -1
        self.val_file = 'val_4.txt'
        self.test_file = 'test_4.txt'


class GinConfig(object):
    def __init__(self):
        # ablations
        self.use_pe = True
        self.use_gin = True
        self.node_resnet_froze = False
        self.background_resnet_froze = False
        self.use_global_image = True
        self.use_classify = False
        self.trans_use_classify = False


        # Learning Rate
        self.lr_backbone = 1e-5
        self.lr = 1e-4

        # Epochs
        self.epochs = 10
        self.lr_drop = 20
        self.start_epoch = 0
        self.weight_decay = 1e-6

        # Backbone
        self.patchEmbedding = "resnet34"
        self.patch_size = 36
        self.backbone = 'gin'
        self.position_embedding = 'sine'
        self.dilation = True
        self.num_layer = 4 # should be 4
        self.num_mlp_per_layer = 2
        self.build_graph_dim = 512
        self.graph_hidden_dim = 512
        self.graph_out_dim = 256
        self.tag_num = [2]
        self.learn_eps = True
        self.max_node = 600

        # Basic
        self.device = 'cuda:0'
        self.seed = 42
        self.batch_size = 16
        self.num_workers = 8
        self.checkpoint = 'xxx.pth'
        self.save_path = './train_log/general/'
        self.clip_max_norm = 0.1

        # Transformer
        self.hidden_dim = 256
        self.pad_token_id = 0
        self.max_position_embeddings = 128
        self.layer_norm_eps = 1e-12
        self.dropout = 0.1
        self.vocab_size = 30522

        self.enc_layers = 3
        self.dec_layers = 3
        self.dim_feedforward = 2048
        self.nheads = 8
        self.pre_norm = True

        # Dataset
        self.dir = 'Report'
        self.limit = -1
        self.val_file = 'val_2.txt'
        self.test_file = 'test_2.txt'
        self.graph_dir = "cell_graphs"