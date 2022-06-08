from easydict import EasyDict as edict

__C = edict()
cfg = __C

# general parameters
__C.general = {}
__C.general.task_name = 'COVID'
__C.general.save_dir = './ckpt'
__C.general.resume_epoch = False  # True
__C.general.seed = 0


# dataset parameters
__C.dataset = {}
__C.dataset.filepath = './dataset/CTMontage/'
__C.dataset.resize = 1024

# # loss
# __C.loss = {}
# __C.loss.name = ['contentVGG','contentmatching', 'styleVGG']#contentVGG, contentMSE,contentmatching, styleVGG


# net
__C.generator = {}
__C.generator.name = 'multipleAdaIN'
__C.generator.ngf = 64
__C.generator.input_nc = 1
__C.generator.output_nc = 1
__C.generator.n_downsampling = 4
__C.generator.n_blocks = 9

__C.discriminator = {}
__C.discriminator.name = 'patchDis'
__C.discriminator.ndf = 64
__C.discriminator.input_nc = 1
__C.discriminator.n_layers = 4

# training parameters
__C.train = {}
__C.vis_on = True
__C.train.gpu_id = '0,1,2,3'
__C.train.manual_seed = 1024
__C.train.epochs = 100
__C.train.batch_size = 4
__C.train.num_threads = 20
__C.train.lr = 0.0001
__C.train.betas = (0.9, 0.999)
__C.train.save_epochs = 100
__C.train.is_test = False

