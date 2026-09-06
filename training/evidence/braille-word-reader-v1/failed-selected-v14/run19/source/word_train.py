"""Train joint cell/dot model with Keras 3 (Torch CUDA or TensorFlow CPU)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from PIL import Image
import keras

from word_data import HEIGHT, WIDTH, augment_patch, make_target, save_json,rotate_patch_180,mirror_patch_horizontal
from word_model import build_model, word_loss, word_relative_loss, image_normalization_spec
from word_decode import calibrate, summarize, DEFAULT_THRESHOLDS
from word_sampling import BalancedFamilyOrder
from word_sources import audit_supplemental_sources, expand_negative_records

TRAINING_SOURCES = ('word_data.py', 'word_model.py', 'word_decode.py', 'word_train.py', 'word_sampling.py', 'word_tail.py', 'word_sources.py')


class Patches(keras.utils.PyDataset):
    def __init__(self, root, records, batch_size, training, seed, camera_augmentation=False, polarity_augmentation=False,half_turn_augmentation=False,illumination_augmentation=False,horizontal_mirror_augmentation=False,balance_source_families=False,partial_geometry_supervision=False):
        super().__init__(workers=2, use_multiprocessing=False, max_queue_size=2)
        self.root, self.records, self.batch_size = root, records, batch_size
        self.training, self.seed, self.epoch = training, seed, 0
        self.camera_augmentation = camera_augmentation
        self.polarity_augmentation = polarity_augmentation
        self.half_turn_augmentation=half_turn_augmentation
        self.illumination_augmentation=illumination_augmentation
        self.horizontal_mirror_augmentation=horizontal_mirror_augmentation
        self.family_sampler=BalancedFamilyOrder(records,seed) if training and balance_source_families else None
        self.partial_geometry_supervision=partial_geometry_supervision
        self.order = np.arange(len(records))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.records)/self.batch_size)

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed+self.epoch*100000+index)
        images, labels = [], []
        for record_index in self.order[index*self.batch_size:(index+1)*self.batch_size]:
            record = self.records[record_index]
            with Image.open(self.root/record["image"]) as source:
                image = source.convert("RGB")
            boxes = np.asarray(record["boxes"], np.float32).reshape(-1, 4)
            masks=np.asarray(record['masks'],int)
            positive_count = len(boxes)
            ignored = np.asarray(record.get('ignore_boxes', []), np.float32).reshape(-1, 4)
            boxes = np.concatenate((boxes, ignored))
            masks = np.concatenate((masks, np.zeros(len(ignored), dtype=int)))
            originally_complete = (boxes[:, 0]>=0) & (boxes[:, 1]>=0) & (boxes[:, 2]<=WIDTH) & (boxes[:, 3]<=HEIGHT)
            if 'originally_complete' in record:
                originally_complete[:positive_count] &= np.asarray(record['originally_complete'],dtype=bool)
            originally_complete[positive_count:] = False
            if self.training:
                # Keep subsequent augmentation draws paired in the orientation
                # ablation, even when the sampled half-turn is disabled.
                half_turn_draw=rng.random() if self.half_turn_augmentation or self.illumination_augmentation or self.horizontal_mirror_augmentation else 1.
                if self.half_turn_augmentation and half_turn_draw<.5:
                    image,boxes,masks=rotate_patch_180(image,boxes,masks)
                if self.horizontal_mirror_augmentation and half_turn_draw<.5:
                    image,boxes,masks=mirror_patch_horizontal(image,boxes,masks)
                pixels, boxes, eligible = augment_patch(image, boxes, rng, self.camera_augmentation,self.polarity_augmentation,self.half_turn_augmentation or self.illumination_augmentation,originally_complete=originally_complete)
            else:
                pixels = np.asarray(image, np.uint8)
                eligible=originally_complete
            images.append(pixels.astype(np.float32)/255.)
            labels.append(make_target(boxes[:positive_count], masks[:positive_count], eligible[:positive_count],
                                      self.partial_geometry_supervision, ignore_boxes=boxes[positive_count:]))
        return np.stack(images), np.stack(labels)

    def on_epoch_end(self):
        self.epoch += 1
        if self.training:
            if self.family_sampler:
                self.order=self.family_sampler.order(self.epoch-1)
            else:
                np.random.default_rng(self.seed+self.epoch).shuffle(self.order)


class WordCheckpoint(keras.callbacks.Callback):
    def __init__(self, root, output, batch_size, retain_epochs=False,train_dataset=None):
        super().__init__()
        self.output, self.batch_size = output, batch_size
        self.retain_epochs = retain_epochs
        self.records = json.loads((root/"validation-words.json").read_text())["records"]
        self.images = np.stack([np.asarray(Image.open(root/row["image"]).convert("RGB"), np.float32)/255. for row in self.records]) if self.records else None
        self.best_rank = None
        self.history = []
        self.train_dataset=train_dataset
        self.sampling_history=[]
        self.sampled_indices=set()

    def on_epoch_begin(self,epoch,logs=None):
        self.epoch_started=time.perf_counter()
        self.sampling_order=None
        self.completed_training_batches=set()
        if self.train_dataset is not None and self.train_dataset.family_sampler:
            self.sampling_order=self.train_dataset.order.copy()
            self.sampling_epoch=self.train_dataset.epoch

    def on_train_batch_end(self,batch,logs=None):
        self.completed_training_batches.add(int(batch))

    def on_epoch_end(self, epoch, logs=None):
        callback_started=time.perf_counter()
        self.model.save(self.output/"last.keras")
        if self.retain_epochs:
            checkpoint_folder=self.output/'retained-epochs'
            checkpoint_folder.mkdir(exist_ok=True)
            (checkpoint_folder/f'epoch-{epoch+1:02d}.keras').write_bytes((self.output/'last.keras').read_bytes())
        if self.images is None:
            return
        save_finished=time.perf_counter()
        predictions = self.model.predict(self.images, batch_size=self.batch_size, verbose=0)
        predict_finished=time.perf_counter()
        thresholds, metrics = calibrate(predictions, self.records)
        calibration_finished=time.perf_counter()
        precision = metrics["accepted_accuracy"] or 0.
        rank = (precision >= .95, metrics["exact_word_accuracy"] if precision >= .95 else precision, metrics["correct"])
        record = {"epoch": epoch+1, "loss": float(logs.get("loss", 0)), "val_loss": float(logs.get("val_loss", 0)), "thresholds": thresholds, "words": metrics}
        if self.sampling_order is not None:
            batches=[self.sampling_order[index*self.batch_size:(index+1)*self.batch_size] for index in sorted(self.completed_training_batches)]
            actual_order=np.concatenate(batches) if batches else np.asarray([],np.int64)
            audit=self.train_dataset.family_sampler.audit(actual_order,self.sampled_indices)
            audit.update(epoch=epoch+1,dataset_epoch=self.sampling_epoch,completed_batches=len(batches),expected_batches=len(self.train_dataset),complete_epoch=len(batches)==len(self.train_dataset))
            self.sampling_history.append(audit)
            save_json(self.output/'sampling-history.json',self.sampling_history)
            record['sampling']=audit
        record['timing_seconds']={'fit_and_patch_validation':callback_started-self.epoch_started,'save_last_and_retained':save_finished-callback_started,'whole_word_predict':predict_finished-save_finished,'whole_word_calibrate':calibration_finished-predict_finished}
        self.history.append(record)
        save_json(self.output/"validation-history.json", self.history)
        brief={key:record[key] for key in ('epoch','loss','val_loss','thresholds','timing_seconds')}
        brief['words']={key:metrics[key] for key in ('total','accepted','correct','exact_word_accuracy','accepted_accuracy')}
        print("WORD_VALIDATION "+json.dumps(brief), flush=True)
        if self.best_rank is None or rank > self.best_rank:
            self.best_rank = rank
            self.model.save(self.output/"best-word.keras")
            np.savez_compressed(self.output/"best-word-validation.npz", outputs=predictions)
            save_json(self.output/"selected-validation.json", record)
        record['timing_seconds']['callback_total']=time.perf_counter()-callback_started
        save_json(self.output/'validation-history.json',self.history)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--full-precision-training", action="store_true", help="Disable cuDNN TF32 training; parity oracle always disables it independently.")
    parser.add_argument("--camera-augmentation", action="store_true", help="Broader anisotropic scale, low contrast and blur on training pixels and boxes only.")
    parser.add_argument('--polarity-augmentation',action='store_true',help='Isotropic scale, bounded float32 photometry and 50 percent illumination polarity inversion; normalized models only.')
    parser.add_argument('--half-turn-augmentation',action='store_true',help='Isotropic scale and bounded photometry; rotate actual pixels180 degrees and permute six-dot masks with probability.5; never invert intensities.')
    parser.add_argument('--illumination-augmentation',action='store_true',help='Isotropic scale and bounded photometry only; preserve source orientation and six-dot masks.')
    parser.add_argument('--horizontal-mirror-augmentation',action='store_true',help='Mirror real pixels horizontally with probability .5; swap dots1/4,2/5,3/6 and preserve vertical relief shading. Cannot combine with half-turn or polarity.')
    parser.add_argument('--balance-source-families',action='store_true',help='Equal positive patch quotas per source family, continuous seeded cycles; retain epoch size and every negative index. No source record is removed.')
    parser.add_argument('--partial-geometry-supervision',action='store_true',help='Supervise full known source boxes for representable cells crossing the canvas; ignore all six dot losses. Unrepresentable centers and previously cut boxes moved wholly inside remain ignored.')
    parser.add_argument('--train-manifest',default='train.json')
    parser.add_argument('--base-train-manifest', help='Preserved original manifest required when appending supplemental sources.')
    parser.add_argument('--supplemental-source-index', action='append', default=[], help='Hash-bound supplemental TRAIN index; original source-split.json stays unchanged.')
    parser.add_argument('--negative-repeat',type=int,default=1,help='Repeat each existing empty-annotation training record this many times; no new negative labels.')
    parser.add_argument('--supplemental-negative-repeat', type=int,
                        help='Required with supplemental sources; preserve base negative-repeat separately.')
    parser.add_argument("--retain-epochs", action="store_true", help="Retain every checkpoint for later frozen validation-only comparison.")
    parser.add_argument("--image-normalization", choices=('raw','channel_gray_std05'), help="Optional weight-free embedded normalization; omitted preserves a resumed model.")
    parser.add_argument("--geometry-loss", choices=('normalized_huber', 'relative_size_huber'),
                        help='Required for resumed training; fresh models default to normalized_huber.')
    args = parser.parse_args(argv)
    if args.resume and args.geometry_loss is None:
        parser.error('--resume requires explicit --geometry-loss to preserve the intended training objective')
    if args.geometry_loss is None:
        args.geometry_loss = 'normalized_huber'
    if bool(args.base_train_manifest) != bool(args.supplemental_source_index):
        parser.error('--base-train-manifest and --supplemental-source-index must be supplied together')
    if bool(args.supplemental_source_index) != (args.supplemental_negative_repeat is not None):
        parser.error('Supplemental source indexes require explicit --supplemental-negative-repeat')
    if args.supplemental_negative_repeat is not None and args.supplemental_negative_repeat < 1:
        parser.error('--supplemental-negative-repeat must be positive')
    if args.negative_repeat<1 or sum((args.camera_augmentation,args.polarity_augmentation,args.half_turn_augmentation,args.illumination_augmentation))>1:
        parser.error('Use positive negative-repeat and at most one optional augmentation profile')
    if args.horizontal_mirror_augmentation and (args.half_turn_augmentation or args.polarity_augmentation):
        parser.error('Horizontal mirroring cannot combine with half-turn or polarity augmentation')
    return args


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = args.output/"source"
    provenance.mkdir(exist_ok=True)
    for name in TRAINING_SOURCES:
        (provenance/name).write_bytes((Path(__file__).parent/name).read_bytes())
    keras.utils.set_random_seed(args.seed)
    device = "cpu"
    if keras.backend.backend() == "torch":
        import torch
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError("Torch training requires the authorized CUDA device")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = not args.full_precision_training
        torch.set_float32_matmul_precision("highest")
        device = torch.cuda.get_device_name(0)
    else:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(6)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    train_rows = json.loads((args.data/args.train_manifest).read_text())
    supplemental_audit = audit_supplemental_sources(train_rows, args.data, args.base_train_manifest, args.supplemental_source_index)
    val_rows = json.loads((args.data/"validation.json").read_text())
    unique_train_count=len(train_rows)
    negatives=[row for row in train_rows if not row['boxes']]
    train_rows, negative_expansion = expand_negative_records(train_rows, args.negative_repeat,
                                                            supplemental_audit['supplemental_sources'] if supplemental_audit else (),
                                                            args.supplemental_negative_repeat)
    if args.limit_train:
        train_rows = train_rows[:args.limit_train]
    train = Patches(args.data, train_rows, args.batch_size, True, args.seed, args.camera_augmentation,args.polarity_augmentation,args.half_turn_augmentation,args.illumination_augmentation,args.horizontal_mirror_augmentation,args.balance_source_families,args.partial_geometry_supervision)
    validation = Patches(args.data, val_rows, args.batch_size, False, args.seed+1,partial_geometry_supervision=args.partial_geometry_supervision)
    model = keras.models.load_model(args.resume, compile=False) if args.resume else build_model()
    if args.image_normalization is not None:
        initial_weights=model.get_weights()
        model=build_model(weights=None,image_normalization=args.image_normalization)
        model.set_weights(initial_weights)
        del initial_weights
    if (args.polarity_augmentation or args.half_turn_augmentation or args.illumination_augmentation) and image_normalization_spec(model)['kind']=='none':
        raise ValueError('Polarity profile requires embedded per-image normalization')
    model.compile(optimizer=keras.optimizers.Adam(args.learning_rate, clipnorm=10.), loss=word_relative_loss if args.geometry_loss=='relative_size_huber' else word_loss, jit_compile=False)
    config = {"backend": keras.backend.backend(), "keras_version": keras.__version__, "device": device, "architecture": "MobileNetV2 alpha0.5 ImageNet, FPN stride16/8/4, joint sigmoid center/box/dots", "input_shape": [1, HEIGHT, WIDTH, 3], "normalization": "RGB float32 /255; internal Rescaling(2,-1)", "output_shape": [1, HEIGHT//4, WIDTH//4, 11], "output_channels": ["objectness", "dx", "dy", "width", "height", "dot1", "dot2", "dot3", "dot4", "dot5", "dot6"], "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "seed": args.seed, "train_patches": len(train_rows), "validation_patches": len(val_rows), "resume": str(args.resume) if args.resume else None, "physical_camera_validated": False,
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in TRAINING_SOURCES},
              "data_sha256": {name: hashlib.sha256((args.data/name).read_bytes()).hexdigest() for name in ("source-split.json", "train.json", "validation.json", "validation-words.json")}}
    pretrained = Path.home()/".keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224_no_top.h5"
    if pretrained.is_file():
        config["imagenet_weights_sha256"] = hashlib.sha256(pretrained.read_bytes()).hexdigest()
    if keras.backend.backend() == "torch":
        config.update(torch_version=torch.__version__, cuda_version=torch.version.cuda, tf32_training_enabled=not args.full_precision_training)
    config["camera_augmentation"] = args.camera_augmentation
    config['polarity_augmentation']=args.polarity_augmentation
    config['half_turn_augmentation']=args.half_turn_augmentation
    config['illumination_augmentation']=args.illumination_augmentation
    config['horizontal_mirror_augmentation']=args.horizontal_mirror_augmentation
    config['partial_geometry_supervision']={'enabled':args.partial_geometry_supervision,'complete_cells':'Unchanged objectness, full box and six known bits.','representable_partial_cells':'Only actual canvas-crossing full source boxes with center inside and width/height below canvas size: objectness+full geometry, all six dot labels ignored (-1), no clipped-box substitution.','other_partial_cells':'Ignore objectness region; no geometry/dot target. A previously cropped cell moved fully inside cannot regain a target.'}
    config['sampling']=train.family_sampler.describe() if train.family_sampler else {'strategy':'uniform_patch_shuffle'}
    config['sampling']['keras_batch_shuffle']=not args.balance_source_families
    config['visibility_supervision']='A cell must be complete in the original cached image and after every affine/rotation rasterization. Eligibility cannot become true again after pixels were cut.'
    config['train_manifest']=args.train_manifest
    config['base_train_manifest'] = args.base_train_manifest
    config['supplemental_source_indexes'] = args.supplemental_source_index
    config['supplemental_source_audit'] = supplemental_audit
    if supplemental_audit:
        config['data_sha256'][args.base_train_manifest] = supplemental_audit['preserved_base_manifest_sha256']
        for index in args.supplemental_source_index:
            config['data_sha256'][index] = hashlib.sha256((args.data/index).read_bytes()).hexdigest()
    config['data_sha256'][args.train_manifest]=hashlib.sha256((args.data/args.train_manifest).read_bytes()).hexdigest()
    config['unique_train_patches']=unique_train_count
    config['negative_repeat']=args.negative_repeat
    config['supplemental_negative_repeat'] = args.supplemental_negative_repeat
    config['negative_expansion'] = negative_expansion
    config['unique_negative_patches']=len(negatives)
    if args.polarity_augmentation or args.half_turn_augmentation or args.illumination_augmentation:
        config['augmentation_protocol']={'isotropic_zoom':[.8,1.7],'polarity_inversion_probability':.5,'contrast':[.3,1.6],'target_channel_mean':[.35,.8],'gaussian_blur_max_pixels':2.,'generated_low_contrast_fallback_std':.01,'photometric_clamp':'once after floating-point contrast/exposure','source_and_annotation_filtering':False}
        if args.half_turn_augmentation:
            config['augmentation_protocol'].update(polarity_inversion_probability=0.,half_turn_probability=.5,half_turn_bit_mapping=[6,5,4,3,2,1],front_side_annotation_identity='Preserve each source row; rotate all real pixels and boxes together, permute occupied six-dot positions; do not relabel reverse-side impressions.')
        if args.illumination_augmentation:
            config['augmentation_protocol'].update(polarity_inversion_probability=0.,half_turn_probability=0.,front_side_annotation_identity='Preserve source orientation, cell masks, and front-side annotation membership.',random_draw_protocol='Consume the same initial half-turn draw as the rotation profile, but do not rotate; subsequent augmentation RNG draws stay paired.')
    if args.horizontal_mirror_augmentation:
        config.setdefault('augmentation_protocol',{}).update(horizontal_mirror_probability=.5,horizontal_mirror_bit_mapping=[4,5,6,1,2,3],front_side_annotation_identity='Preserve every source annotation; mirror real pixels and boxes together, swap left/right occupied positions; preserve vertical shading and do not invert pixel intensities.',random_draw_protocol='Reuse the initial orientation draw; subsequent illumination RNG draws stay paired with the illumination-only profile.')
    config["retain_epochs"] = args.retain_epochs
    config['geometry_loss']=args.geometry_loss
    config['image_normalization']='channel_gray_std05' if any(layer.name=='camera_normalization' for layer in model.layers) else 'raw'
    config['imageNormalization']=image_normalization_spec(model)
    backbone=model.get_layer('mobilenetv2_half')
    strides = sorted({HEIGHT//int(output.shape[1]) for output in backbone.outputs}, reverse=True)
    config['architecture'] = 'MobileNetV2 alpha0.5 ImageNet, FPN stride'+('/'.join(map(str,strides)))+', joint sigmoid center/box/dots'
    config['feature_pyramid_strides'] = strides
    config['parameters'] = model.count_params()
    batchnorm_layers=[layer for layer in backbone.layers if isinstance(layer,keras.layers.BatchNormalization)]
    backbone_graph=next(layer for layer in model.get_config()['layers'] if layer['name']=='mobilenetv2_half')
    config['batch_normalization']={
        'layers':len(batchnorm_layers),
        'trainable_affine_layers':sum(layer.trainable for layer in batchnorm_layers),
        'momentum_values':sorted(set(float(layer.momentum) for layer in batchnorm_layers)),
        'epsilon_values':sorted(set(float(layer.epsilon) for layer in batchnorm_layers)),
        'backbone_call_training_keywords':[node.get('kwargs',{}).get('training','inherit_outer_model') for node in backbone_graph['inbound_nodes']],
        'evaluation_mode':'Keras predict/test and the export serving function call the outer model with training=False; stored moving moments are used.',
    }
    if config['image_normalization']=='channel_gray_std05':
        config['normalization']='RGB float32 /255; internal per-image per-channel mean and population std (ddof=0, std floor .001), mean standardized channels, .5+.05*gray clipped [0,1], repeat RGB, then Rescaling(2,-1)'
    config["resume_sha256"] = hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None
    save_json(args.output/"training-config.json", config)
    print(json.dumps(config), flush=True)
    start = time.time()
    history = model.fit(train, validation_data=validation, epochs=args.epochs, verbose=2, shuffle=not args.balance_source_families, callbacks=[WordCheckpoint(args.data, args.output, args.batch_size,args.retain_epochs,train), keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=4, factor=.5, min_lr=1e-5), keras.callbacks.TerminateOnNaN()])
    save_json(args.output/"history.json", {"elapsed_seconds": time.time()-start, "history": {k: [float(v) for v in values] for k, values in history.history.items()}})


if __name__ == "__main__":
    main()
