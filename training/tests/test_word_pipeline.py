"""Meaningful geometry, six-bit, and incomplete-word regressions without a model."""
import sys
import unittest
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from word_data import WIDTH, HEIGHT, GRID_W, GRID_H, make_target, word_groups, boundary_guard, wrapped_line_starts, affine_patch, augment_patch,rotate_patch_180,mirror_patch_horizontal
from word_decode import decode, DEFAULT_THRESHOLDS, summarize, prepare_calibration_outputs
from word_test_predictions import center_predictions


def output_word(masks):
    width = min(28, 400/(len(masks)*1.4))
    start = (WIDTH-width*len(masks)*1.4)/2
    boxes = np.asarray([[start+i*width*1.4, 65, start+i*width*1.4+width, 115] for i in range(len(masks))])
    return center_predictions(make_target(boxes, np.asarray(masks)))


class WordPipelineTests(unittest.TestCase):
    def test_partial_geometry_retains_full_source_box_and_ignores_all_bits(self):
        target=make_target(np.asarray([[200.,-20.,240.,60.]]),np.asarray([1]),partial_geometry_supervision=True)
        ys,xs=np.where(target[...,0]==1)
        self.assertEqual((ys.tolist(),xs.tolist()),([5],[55]))
        row=target[5,55]
        self.assertEqual(row[5:].tolist(),[-1]*6)
        self.assertAlmostEqual(float(row[3]*WIDTH),40.,places=5)
        self.assertAlmostEqual(float(row[4]*HEIGHT),80.,places=5)
        prediction=center_predictions(np.maximum(target,0))
        prediction[5,55,5:]=.5
        self.assertEqual(decode(prediction)['reason'],'clipped')

    def test_partial_geometry_cannot_label_unrepresentable_or_restored_boxes(self):
        boxes=np.asarray([[100.,-100.,140.,20.],[150.,-100.,190.,200.],[200.,50.,240.,90.]])
        target=make_target(boxes,np.asarray([1,3,9]),np.asarray([False,False,False]),partial_geometry_supervision=True)
        self.assertEqual(int((target[...,0]==1).sum()),0)
        self.assertGreater(int((target[...,0]<0).sum()),0)

    def test_complete_cell_wins_a_grid_collision_with_partial_geometry(self):
        boxes=np.asarray([[200.,-20.,240.,60.],[210.,10.,230.,30.]])
        target=make_target(boxes,np.asarray([63,1]),partial_geometry_supervision=True)
        self.assertEqual(int((target[...,0]==1).sum()),1)
        self.assertEqual(target[5,55,5:].tolist(),[1,0,0,0,0,0])

    def test_horizontal_mirror_exact_pixels_geometry_all_masks_and_inverse(self):
        pixels=np.random.default_rng(12).integers(0,256,(HEIGHT,WIDTH,3),dtype=np.uint8)
        boxes=np.tile(np.asarray([[-4.,12.5,30.,50.25],[120.5,40.,170.25,92.]]),(32,1))
        masks=np.arange(64,dtype=np.int64)
        image,mirrored,remapped=mirror_patch_horizontal(Image.fromarray(pixels),boxes,masks)
        np.testing.assert_array_equal(np.asarray(image),pixels[:,::-1])
        np.testing.assert_array_equal(mirrored[:,[1,3]],boxes[:,[1,3]])
        self.assertEqual(remapped[[1,2,4,8,16,32]].tolist(),[8,16,32,1,2,4])
        image,restored_boxes,restored_masks=mirror_patch_horizontal(image,mirrored,remapped)
        np.testing.assert_array_equal(np.asarray(image),pixels)
        np.testing.assert_array_equal(restored_boxes,boxes)
        np.testing.assert_array_equal(restored_masks,masks)

    def test_horizontal_mirror_preserves_partial_identity_and_vertical_shading(self):
        pixels=np.full((HEIGHT,WIDTH,3),128,np.uint8)
        pixels[30:35,20:25]=240
        pixels[35:40,20:25]=20
        boxes=np.asarray([[-8.,20.,30.,70.],[100.,50.,120.,90.]])
        image,mirrored,masks=mirror_patch_horizontal(Image.fromarray(pixels),boxes,np.asarray([1,3]))
        self.assertEqual(image.getpixel((WIDTH-23,32)),(240,240,240))
        self.assertEqual(image.getpixel((WIDTH-23,37)),(20,20,20))
        target=make_target(mirrored,masks,np.asarray([False,True]))
        self.assertEqual(int((target[...,0]==1).sum()),1)
        self.assertEqual(masks.tolist(),[8,24])
        self.assertTrue(mirrored[0,2]>WIDTH)

    def test_half_turn_is_exact_involution_for_pixels_boxes_all_masks(self):
        pixels=np.random.default_rng(5).integers(0,256,(HEIGHT,WIDTH,3),dtype=np.uint8)
        boxes=np.tile(np.asarray([[-4.,12.5,30.,50.25],[120.5,40.,170.25,92.]]),(32,1))
        masks=np.arange(64,dtype=np.int64)
        image,first,remapped=rotate_patch_180(Image.fromarray(pixels),boxes,masks)
        image,second,restored=rotate_patch_180(image,first,remapped)
        self.assertTrue(np.array_equal(np.asarray(image),pixels))
        self.assertTrue(np.array_equal(second,boxes))
        self.assertTrue(np.array_equal(restored,masks))
        self.assertEqual(remapped[[1,2,4,8,16,32]].tolist(),[32,16,8,4,2,1])

    def test_half_turn_preserves_cut_cell_identity_and_pixel_intensity(self):
        pixels=np.full((HEIGHT,WIDTH,3),210,np.uint8)
        pixels[20:30,30:40]=65
        boxes=np.asarray([[-8.,50.,20.,90.],[100.,60.,120.,100.]])
        masks=np.asarray([1,3]);complete=np.asarray([False,True])
        image,rotated,remapped=rotate_patch_180(Image.fromarray(pixels),boxes,masks)
        self.assertEqual(np.unique(np.asarray(image)).tolist(),[65,210])
        target=make_target(rotated,remapped,complete)
        self.assertEqual(int((target[...,0]==1).sum()),1)
        self.assertEqual(remapped.tolist(),[32,48])
        self.assertTrue(rotated[0,2]>WIDTH)

    def test_generated_low_contrast_does_not_erase_positive_source(self):
        class FaintContrast:
            def random(self):return .9
            def uniform(self,low,high):return low
        source=np.full((HEIGHT,WIDTH,3),128,np.uint8)
        source[:,WIDTH//2:]=140
        boxes=np.asarray([[100.,50.,120.,90.]])
        pixels,transformed,eligible=augment_patch(Image.fromarray(source),boxes,FaintContrast(),polarity_augmentation=True)
        self.assertTrue(np.array_equal(pixels,source))
        self.assertTrue(np.array_equal(transformed,boxes))
        self.assertEqual(eligible.tolist(),[True])

    def test_intermediate_clipping_cannot_resurrect_an_erased_dot_label(self):
        class CropThenRotate:
            def __init__(self):self.random_index=0;self.translation_index=0
            def random(self):
                self.random_index+=1
                return 0. if self.random_index<=2 else 1.
            def uniform(self,low,high):
                if (low,high)==(-.04,.04):
                    self.translation_index+=1
                    return -3/WIDTH if self.translation_index==1 else 0.
                if (low,high)==(-4,4):return 3.8
                return 1.
        pixels=np.zeros((HEIGHT,WIDTH,3),np.uint8)
        pixels[125:127,1:3]=255
        boxes=np.asarray([[1.,125.,17.,155.]])
        original=np.asarray([True])
        transformed,final,eligible=augment_patch(Image.fromarray(pixels),boxes,CropThenRotate(),camera_augmentation=True,originally_complete=original)
        # The affine step loses the entire marker; subsequent rotation returns
        # the complete annotation rectangle to the canvas, but no real pixels.
        self.assertEqual(int((transformed[...,0]>127).sum()),0)
        self.assertTrue(np.all(final[:,:2]>=0))
        self.assertTrue(np.all(final[:,2:]<=[WIDTH,HEIGHT]))
        self.assertEqual(eligible.tolist(),[False])
        self.assertEqual(original.tolist(),[True])
        old_target=make_target(final,np.asarray([1]),original)
        target=make_target(final,np.asarray([1]),eligible)
        self.assertEqual(int((old_target[...,0]==1).sum()),1)
        self.assertEqual(int((target[...,0]==1).sum()),0)
        self.assertGreater(int((target[...,0]<0).sum()),0)

    def test_zoom_out_preserves_original_partial_flags_and_cell_order(self):
        class ZoomOut:
            def __init__(self):self.random_index=0
            def random(self):self.random_index+=1;return 0. if self.random_index==1 else 1.
            def uniform(self,low,high):
                if (low,high)==(.85,1.35):return .85
                if (low,high)==(-.04,.04):return 0.
                return 1.
        boxes=np.asarray([[-4.,80.,16.,110.],[120.,70.,150.,100.],[200.,70.,230.,100.]])
        image=Image.new('RGB',(WIDTH,HEIGHT),'white')
        _,transformed,eligible=augment_patch(image,boxes,ZoomOut(),camera_augmentation=True,originally_complete=np.asarray([False,False,True]))
        self.assertTrue(np.all(transformed[:,:2]>=0))
        self.assertTrue(np.all(transformed[:,2:]<=[WIDTH,HEIGHT]))
        self.assertEqual(eligible.tolist(),[False,False,True])
        target=make_target(transformed,np.asarray([1,3,9]),eligible)
        centers=target[target[...,0]==1]
        self.assertEqual(len(centers),1)
        self.assertEqual(centers[0,5:].tolist(),[1,0,0,1,0,0])

    def test_rotation_updates_partial_eligibility_without_an_affine_step(self):
        class RotateOnly:
            def __init__(self):self.random_index=0
            def random(self):self.random_index+=1;return 0. if self.random_index==1 else 1.
            def uniform(self,low,high):return 3.8 if (low,high)==(-4,4) else 1.
        _,_,eligible=augment_patch(Image.new('RGB',(WIDTH,HEIGHT),'white'),np.asarray([[1.,1.,17.,31.]]),RotateOnly())
        self.assertEqual(eligible.tolist(),[False])

    def test_cached_minimum_score_filter_preserves_every_decoder_result(self):
        rng=np.random.default_rng(42)
        predictions=np.stack([output_word([13,1,13,1]),output_word([5,1,5,1,5]),rng.random((GRID_H,GRID_W,11),dtype=np.float32)])
        predictions[0,0,0,0]=predictions[0,0,1,0]=.8
        prepared=prepare_calibration_outputs(predictions)
        for original,cached in zip(predictions,prepared):
            for proposal in (.15,.25,.35):
                threshold=dict(DEFAULT_THRESHOLDS,proposalScore=proposal)
                self.assertEqual(decode(original,threshold),decode(cached,threshold))

    def test_affine_augmentation_transforms_pixels_and_boxes_together(self):
        image = Image.new('RGB', (WIDTH, HEIGHT), 'white')
        ImageDraw.Draw(image).rectangle((100, 60, 140, 100), fill='black')
        transformed, boxes = affine_patch(image, np.asarray([[100, 60, 140, 100]]), .5, 1.2, 10, -4)
        np.testing.assert_allclose(boxes, [[180, 48.8, 200, 96.8]])
        self.assertEqual(transformed.getpixel((190, 72)), (0, 0, 0))
        self.assertEqual(transformed.getpixel((160, 72)), (255, 255, 255))

    def test_offscreen_augmented_cells_do_not_mask_real_background(self):
        target = make_target(np.asarray([[-50, 50, -20, 90], [500, 50, 530, 90]]), np.asarray([1, 1]))
        self.assertTrue(np.all(target==0))

    def test_zoom_out_cannot_restore_originally_cropped_dots(self):
        target = make_target(np.asarray([[100, 50, 130, 90]]), np.asarray([3]), np.asarray([False]))
        self.assertEqual(int((target[..., 0]==1).sum()), 0)
        self.assertLess(target[16, 28, 0], 0)

    def test_repeated_letters_are_preserved(self):
        self.assertEqual(decode(output_word([13, 1, 13, 1]))["text"], "mama")
        self.assertEqual(decode(output_word([5, 1, 5, 1, 5]))["text"], "kakak")

    def test_one_and_twelve_letters(self):
        self.assertEqual(decode(output_word([1]))["text"], "a")
        self.assertEqual(decode(output_word([1]*12))["text"], "a"*12)
        self.assertEqual(decode(output_word([1]*13))["reason"], "too_many_characters")

    def test_unsupported_six_bit_pattern_rejects_whole_word(self):
        self.assertEqual(decode(output_word([1, 63, 1]))["reason"], "unsupported_pattern")
        self.assertEqual(decode(output_word([1, 63, 1]))["text"], "")

    def test_partial_cells_are_ignored_not_trained_as_background(self):
        target = make_target(np.asarray([[-8, 50, 20, 90], [200, 50, 220, 90]]), np.asarray([1, 3]))
        self.assertLess(target[16, 1, 0], 0)
        self.assertEqual(int((target[..., 0]==1).sum()), 1)
        center = target[target[..., 0]==1][0]
        self.assertEqual(center[5:].tolist(), [1, 1, 0, 0, 0, 0])

    def test_uncertain_retained_cell_cannot_be_dropped(self):
        output = output_word([13, 1, 13, 1])
        row, column = np.argwhere(output[..., 0]==1)[1]
        output[row, column, 5] = .5
        self.assertEqual(decode(output)["text"], "")
        self.assertEqual(decode(output)["reason"], "low_confidence")

    def test_gap_rejects_missing_middle_cell(self):
        boxes = np.asarray([[80, 70, 100, 110], [270, 70, 290, 110]])
        self.assertEqual(decode(center_predictions(make_target(boxes, np.asarray([1, 1]))))["reason"], "gap")

    def test_multiple_rows_and_clipped_cells_reject(self):
        boxes = np.asarray([[150, 10, 170, 50], [200, 130, 220, 170]])
        self.assertEqual(decode(center_predictions(make_target(boxes, np.asarray([1, 1]))))["reason"], "multiple_rows")
        boxes = np.asarray([[0, 70, 20, 110]])
        self.assertEqual(decode(center_predictions(make_target(boxes, np.asarray([1]))))["reason"], "clipped")

    def test_empty_malformed_and_nonfinite_output(self):
        self.assertEqual(decode(np.zeros((GRID_H, GRID_W, 11)))["reason"], "no_grid")
        self.assertEqual(decode(np.zeros((GRID_H, GRID_W, 5)))["reason"], "malformed")
        self.assertEqual(decode(np.full((GRID_H, GRID_W, 11), np.nan))["reason"], "malformed")

    def test_proposal_threshold_uses_javascript_number_precision(self):
        output = output_word([1]).astype(np.float32)
        output[..., 0] *= np.float32(.35)
        thresholds = dict(DEFAULT_THRESHOLDS, proposalScore=.35, acceptScore=.35)
        self.assertLess(float(np.float32(.35)), .35)
        self.assertEqual(decode(output, thresholds)["reason"], "no_grid")
        output[..., 0] = np.where(output[..., 0]>.34, np.nextafter(np.float32(.35), np.float32(1)), 0)
        self.assertEqual(decode(output, thresholds)["text"], "a")

    def test_word_boundary_groups_do_not_slice_cell_runs(self):
        boxes = np.asarray([[x, 50, x+20, 90] for x in (10, 38, 66, 122, 150)])
        self.assertEqual(word_groups(boxes), [[0, 1, 2], [3, 4]])

    def test_narrow_annotation_boxes_do_not_split_normal_center_pitch(self):
        boxes = np.asarray([[cx-8.5, 50, cx+8.5, 85] for cx in (276.5, 311, 349, 386, 421)])
        self.assertEqual(word_groups(boxes), [[0, 1, 2, 3, 4]])

    def test_boundary_guard_checks_neighbors_outside_incorrect_line_cluster(self):
        boxes = np.asarray([[402, 315, 413, 334], [421, 313, 432, 332], [439, 313, 450, 332], [457, 313, 468, 332]])
        self.assertEqual(boundary_guard(boxes, [0], 1024), "uncertain_boundary_neighbor")
        self.assertIsNone(boundary_guard(boxes, [0, 1, 2, 3], 1024))

    def test_boundary_guard_rejects_unverifiable_photo_edge(self):
        self.assertEqual(boundary_guard(np.asarray([[990, 50, 1018, 90]]), [0], 1024), "uncertain_image_edge_boundary")

    def test_line_wrap_suffix_is_not_a_complete_word(self):
        boxes = np.asarray([[x, y, x+20, y+35] for x, y in ((20, 50), (50, 50), (80, 50), (20, 110), (50, 110))])
        self.assertEqual(wrapped_line_starts(boxes, np.asarray([1, 3, 36, 13, 1])), {3})

    def test_selection_ignores_other_rows_but_never_cut_cells(self):
        boxes = np.asarray([[100, 10, 120, 40], [200, 75, 220, 115], [250, 150, 270, 180]])
        output = center_predictions(make_target(boxes, np.asarray([3, 1, 9])))
        self.assertEqual(decode(output, selection=[0, .34, 1, .66])["text"], "a")
        self.assertEqual(decode(output, selection=[0, .5, 1, .66])["reason"], "clipped")

    def test_rejections_count_as_wrong_words(self):
        metrics = summarize([output_word([13, 1]), np.zeros((GRID_H, GRID_W, 11))], [{"expected": "ma"}, {"expected": "ma"}])
        self.assertEqual(metrics["exact_word_accuracy"], .5)
        self.assertEqual(metrics["accepted_accuracy"], 1.)

    def test_indicators_retain_physical_cells_and_decode_case_or_integer(self):
        for masks, text in [([32, 1, 3], 'Ab'), ([32, 32, 1, 3], 'AB'), ([60, 26, 26, 1], '001'), ([32, 1] * 12, 'A' * 12)]:
            with self.subTest(text=text):
                result = decode(output_word(masks))
                self.assertEqual(result['text'], text)
                self.assertEqual(len(result['cells']), len(masks))
                self.assertEqual(''.join(cell['output'] for cell in result['cells']), text)
        self.assertEqual(decode(output_word([1] * 25))['reason'], 'too_many_cells')

    def test_uncertain_indicator_rejects_entire_token(self):
        output = output_word([60, 1, 3])
        row, column = np.argwhere(output[..., 0] >= .99)[0]
        output[row, column, 5] = .5
        result = decode(output)
        self.assertEqual(result['text'], '')
        self.assertEqual(result['reason'], 'low_confidence')


if __name__ == "__main__":
    unittest.main()
