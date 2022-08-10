import time
from enum import IntEnum
from xlib.face import FRect
from xlib.image import ImageProcessor
from xlib.face import ELandmarks2D, FLandmarks2D, FPose
from modelhub import onnx as onnx_models
import numpy as np
import cv2
from . import backend

from xlib.face import FRect, FLandmarks2D, FPose


class BackendFaceSwapInfo:
    def __init__(self):
        self.image_name = None
        self.face_urect : FRect = None
        self.face_pose : FPose = None
        self.face_ulmrks : FLandmarks2D = None

        self.face_resolution : int = None
        self.face_align_image_name : str = None
        self.face_align_mask_name : str = None
        self.face_align_lmrks_mask_name : str = None
        self.face_anim_image_name : str = None
        self.face_swap_image_name : str = None
        self.face_swap_mask_name : str = None

        self.image_to_align_uni_mat = None
        self.face_align_ulmrks : FLandmarks2D = None

    def __getself__(self):
        return self.__dict__.copy()

    def __setself__(self, d):
        self.__init__()
        self.__dict__.update(d)


class _ResolutionType(IntEnum):
    RES_320x240 = 0
    RES_640x480 = 1
    RES_720x480 = 2
    RES_1280x720 = 3
    RES_1280x960 = 4
    RES_1366x768 = 5
    RES_1920x1080 = 6

class _DriverType(IntEnum):
    COMPATIBLE = 0
    DSHOW = 1
    MSMF = 2
    GSTREAMER = 3

class _RotationType(IntEnum):
    ROTATION_0 = 0
    ROTATION_90 = 1
    ROTATION_180 = 2
    ROTATION_270 = 3


_ResolutionType_wh = {_ResolutionType.RES_320x240: (320,240),
                      _ResolutionType.RES_640x480: (640,480),
                      _ResolutionType.RES_720x480: (720,480),
                      _ResolutionType.RES_1280x720: (1280,720),
                      _ResolutionType.RES_1280x960: (1280,960),
                      _ResolutionType.RES_1366x768: (1366,768),
                      _ResolutionType.RES_1920x1080: (1920,1080),
                      }


class Camera():
    def __init__(self) -> None:
            self.vcap = None
            driver = _DriverType.DSHOW

            cv_api = {_DriverType.COMPATIBLE: cv2.CAP_ANY,
                      _DriverType.DSHOW: cv2.CAP_DSHOW,
                      _DriverType.MSMF: cv2.CAP_MSMF,
                      _DriverType.GSTREAMER: cv2.CAP_GSTREAMER,
                      }[driver]

            vcap = cv2.VideoCapture(0)
            
            if vcap.isOpened():
                self.vcap = vcap
                w, h = _ResolutionType_wh[0]
                vcap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)


    def on_tick(self):
        ret, img = self.vcap.read()
        if ret:
            ip = ImageProcessor(img)
            ip.ch(3).to_uint8()
            w, h = _ResolutionType_wh[0]
            ip.fit_in(TW=w)
            img = ip.get_image('HWC')
            return img



class DetectorType(IntEnum):
    CENTER_FACE = 0
    S3FD = 1
    YOLOV5 = 2

DetectorTypeNames = ['CenterFace', 'S3FD', 'YoloV5']

class FaceSortBy(IntEnum):
    LARGEST = 0
    DIST_FROM_CENTER = 1
    LEFT_RIGHT = 2
    RIGHT_LEFT = 3
    TOP_BOTTOM = 4
    BOTTOM_TOP = 5


FaceSortByNames = ['@FaceDetector.LARGEST', '@FaceDetector.DIST_FROM_CENTER',
                   '@FaceDetector.LEFT_RIGHT', '@FaceDetector.RIGHT_LEFT',
                   '@FaceDetector.TOP_BOTTOM', '@FaceDetector.BOTTOM_TOP' ]





class FacDetetor():

    detector_type = DetectorType.YOLOV5


    def __init__(self,device="CPU"):
            if self.detector_type == DetectorType.CENTER_FACE:
                self.CenterFace = onnx_models.CenterFace(device)
            elif self.detector_type == DetectorType.S3FD:
                self.S3FD = onnx_models.S3FD(device)
            elif self.detector_type == DetectorType.YOLOV5:
                dev_info = onnx_models.YoloV5Face.get_available_devices()
                self.YoloV5Face = onnx_models.YoloV5Face(dev_info[0])


    
    def on_tick(self, frame_image):

                    
                    threshold = 0.5
                    sort_by = FaceSortBy.DIST_FROM_CENTER
                    fixed_window_size = 480
                    max_faces = 0
                    temporal_smoothing = 1

                    swap_info_list = []
                    

                    if frame_image is not None:
                        _,H,W,_ = ImageProcessor(frame_image).get_dims()
                        rects = []
                        
                        if self.detector_type == DetectorType.CENTER_FACE:
                            rects = self.CenterFace.extract (frame_image, threshold=threshold, fixed_window=fixed_window_size)[0]
                        elif self.detector_type == DetectorType.S3FD:
                            rects = self.S3FD.extract (frame_image, threshold=threshold, fixed_window=fixed_window_size)[0]
                        elif self.detector_type == DetectorType.YOLOV5:
                            rects = self.YoloV5Face.extract (frame_image, threshold=threshold, fixed_window=fixed_window_size)[0]

                        # to list of FaceURect
                        rects = [ FRect.from_ltrb( (l/W, t/H, r/W, b/H) ) for l,t,r,b in rects ]

                        # sort
                        if sort_by == FaceSortBy.LARGEST:
                            rects = FRect.sort_by_area_size(rects)
                        elif sort_by == FaceSortBy.DIST_FROM_CENTER:
                            rects = FRect.sort_by_dist_from_2D_point(rects, 0.5, 0.5)
                        elif sort_by == FaceSortBy.LEFT_RIGHT:
                            rects = FRect.sort_by_dist_from_horizontal_point(rects, 0)
                        elif sort_by == FaceSortBy.RIGHT_LEFT:
                            rects = FRect.sort_by_dist_from_horizontal_point(rects, 1)
                        elif sort_by == FaceSortBy.TOP_BOTTOM:
                            rects = FRect.sort_by_dist_from_vertical_point(rects, 0)
                        elif sort_by == FaceSortBy.BOTTOM_TOP:
                            rects = FRect.sort_by_dist_from_vertical_point(rects, 1)

                        if len(rects) != 0:
                            max_faces = max_faces
                            if max_faces != 0 and len(rects) > max_faces:
                                rects = rects[:max_faces]

                            if temporal_smoothing != 1:
                                if len(self.temporal_rects) != len(rects):
                                    self.temporal_rects = [ [] for _ in range(len(rects)) ]

                            for face_id, face_urect in enumerate(rects):
                                if temporal_smoothing != 1:
                                    if not len(self.temporal_rects[face_id]) == 0:
                                        self.temporal_rects[face_id].append( face_urect.as_4pts() )

                                    self.temporal_rects[face_id] = self.temporal_rects[face_id][-temporal_smoothing:]

                                    face_urect = FRect.from_4pts ( np.mean(self.temporal_rects[face_id],0 ) )

                                if face_urect.get_area() != 0:
                                    fsi = BackendFaceSwapInfo()
                                    fsi.face_urect = face_urect
                                    swap_info_list.append(fsi)

                    return swap_info_list

  
class FaceMarker():
    def __init__(self) -> None:
        device = onnx_models.FaceMesh.get_available_devices()
        self.google_facemesh = onnx_models.FaceMesh(device[0])
        self.temporal_smoothing = 1
        self.marker_coverage = 1.4


    def on_tick(self, frame_image, swap_info_list):

        for face_id, fsi in enumerate(swap_info_list):
            if fsi.face_urect is not None:
                # Cut the face to feed to the face marker
                face_image, face_uni_mat = fsi.face_urect.cut(frame_image, self.marker_coverage, 192)
                _,H,W,_ = ImageProcessor(face_image).get_dims()

                
                lmrks = self.google_facemesh.extract(face_image)[0]

                if self.temporal_smoothing != 1:
                     if len(self.temporal_lmrks[face_id]) == 0:
                         self.temporal_lmrks[face_id].append(lmrks)
                     self.temporal_lmrks[face_id] = self.temporal_lmrks[face_id][-self.temporal_smoothing:]
                     lmrks = np.mean(self.temporal_lmrks[face_id],0 )

                fsi.face_pose = FPose.from_3D_468_landmarks(lmrks)

                lmrks = lmrks[...,0:2] / (W,H)

                face_ulmrks = FLandmarks2D.create (ELandmarks2D.L468, lmrks)
                face_ulmrks = face_ulmrks.transform(face_uni_mat, invert=True)
                fsi.face_ulmrks = face_ulmrks


class AlignMode(IntEnum):
    FROM_RECT = 0
    FROM_POINTS = 1
        
class FaceAligner():
    def __init__(self):
        self.align_mode = AlignMode.FROM_POINTS
        self.face_coverage = 2.2
        self.resolution = 224
        self.exclude_moving_parts = True
        self.head_mode = False
        self.freeze_z_rotation = False
        self.x_offset = 0
        self.y_offset = 0

    def on_tick(self,frame_image, face_swap_info_list):
        face_align_lmrks_mask_img = None
        face_align_img = None
        
        for face_id, fsi in enumerate(face_swap_info_list):
            head_yaw = None
            if self.head_mode or self.freeze_z_rotation:
                if fsi.face_pose is not None:
                    head_yaw = fsi.face_pose.as_radians()[1]



            face_ulmrks = fsi.face_ulmrks
            if face_ulmrks is not None:
                fsi.face_resolution = self.resolution
                            
                if self.align_mode == AlignMode.FROM_RECT:
                    face_align_img, uni_mat = fsi.face_urect.cut(frame_image, coverage= self.face_coverage, output_size=self.resolution,
                                                                     x_offset=self.x_offset, y_offset=self.y_offset)


                elif self.align_mode == AlignMode.FROM_POINTS:
                    face_align_img, uni_mat = face_ulmrks.cut(frame_image, self.face_coverage, self.resolution,
                                                            exclude_moving_parts=self.exclude_moving_parts,
                                                            head_yaw=head_yaw,
                                                            x_offset=self.x_offset,
                                                            y_offset=self.y_offset-0.08, freeze_z_rotation=self.freeze_z_rotation)                                                            
            

                #fsi.face_align_image_name = f'{frame_image_name}_{face_id}_aligned'
                fsi.image_to_align_uni_mat = uni_mat
                fsi.face_align_ulmrks = face_ulmrks.transform(uni_mat)
                #bcd.set_image(fsi.face_align_image_name, face_align_img)


                # Due to FaceAligner is not well loaded, we can make lmrks mask here
                face_align_lmrks_mask_img = fsi.face_align_ulmrks.get_convexhull_mask( face_align_img.shape[:2], color=(255,), dtype=np.uint8)
        
        return face_align_lmrks_mask_img , face_align_img
                #fsi.face_align_lmrks_mask_name = f'{frame_image_name}_{face_id}_aligned_lmrks_mask'
                #bcd.set_image(fsi.face_align_lmrks_mask_name, face_align_lmrks_mask_img)

from modelhub import DFLive
from pathlib import Path
from xlib.python import all_is_not_None

class FaceSwapper():
    def __init__(self):
        self.model = DFLive.get_available_models_info(Path("/data/dfm_models/"))[2]
        self.device = DFLive.get_available_devices()[0] 
        self.dfm_model_initializer = DFLive.DFMModel_from_info(self.model, self.device)
        self.dfm_model = None
        self.swap_all_faces = False
        self.face_id = 0
        self.morph_factor = 0
        self.presharpen_amount = 0
        self.pre_gamma_red = 1
        self.pre_gamma_green = 1
        self.pre_gamma_blue = 1
        self.post_gamma_red = 0.9
        self.post_gamma_blue = 1.1
        self.post_gamma_green = 1
        self.two_pass = False



    def on_tick(self,face_align_image, face_swap_info_list):

        face_align_mask_img = None 
        celeb_face = None 
        celeb_face_mask_img = None

        if self.dfm_model_initializer is not None:
            events = self.dfm_model_initializer.process_events()

           

            if events.prev_status_downloading:
                 print("prev_status_downloading")


            if events.new_status_downloading:
                print("new_status_downloading")


            elif events.new_status_initialized:
                print("new_status_initialized")
                self.dfm_model = events.dfm_model
                self.dfm_model_initializer = None

                model_width, model_height = self.dfm_model.get_input_res()



                dfm_model = self.dfm_model

        if all_is_not_None(self.dfm_model, face_swap_info_list):

            for i, fsi in enumerate(face_swap_info_list):
                        if not self.swap_all_faces and self.face_id != i:
                            continue

                        #face_align_image = bcd.get_image(fsi.face_align_image_name)
                        if face_align_image is not None:

                            pre_gamma_red = self.pre_gamma_red
                            pre_gamma_green = self.pre_gamma_green
                            pre_gamma_blue = self.pre_gamma_blue
                            post_gamma_red = self.post_gamma_red
                            post_gamma_blue = self.post_gamma_blue
                            post_gamma_green = self.post_gamma_green

                            fai_ip = ImageProcessor(face_align_image)
                            if self.presharpen_amount != 0:
                                fai_ip.gaussian_sharpen(sigma=1.0, power=self.presharpen_amount)

                            if pre_gamma_red != 1.0 or pre_gamma_green != 1.0 or pre_gamma_blue != 1.0:
                                fai_ip.gamma(pre_gamma_red, pre_gamma_green, pre_gamma_blue)
                            face_align_image = fai_ip.get_image('HWC')

                            celeb_face, celeb_face_mask_img, face_align_mask_img = self.dfm_model.convert(face_align_image, morph_factor=self.morph_factor)
                            celeb_face, celeb_face_mask_img, face_align_mask_img = celeb_face[0], celeb_face_mask_img[0], face_align_mask_img[0]

                            if self.two_pass:
                                celeb_face, celeb_face_mask_img, _ = self.dfm_model.convert(celeb_face, morph_factor=self.morph_factor)
                                celeb_face, celeb_face_mask_img = celeb_face[0], celeb_face_mask_img[0]

                            if post_gamma_red != 1.0 or post_gamma_blue != 1.0 or post_gamma_green != 1.0:
                                celeb_face = ImageProcessor(celeb_face).gamma(post_gamma_red, post_gamma_blue, post_gamma_green).get_image('HWC')

                            fsi.face_align_mask_name = f'{fsi.face_align_image_name}_mask'
                            fsi.face_swap_image_name = f'{fsi.face_align_image_name}_swapped'
                            fsi.face_swap_mask_name  = f'{fsi.face_swap_image_name}_mask'

                            
                            

                            # bcd.set_image(fsi.face_align_mask_name, face_align_mask_img)
                            # bcd.set_image(fsi.face_swap_image_name, celeb_face)
                            # bcd.set_image(fsi.face_swap_mask_name, celeb_face_mask_img)

        return (face_align_mask_img, celeb_face, celeb_face_mask_img)

class FrameAdjuster():

    def __init__(self):
        self.median_blur_per = 60.0
        self.degrade_bicubic_per = 60.0

    def on_tick(self, frame_image):


                if frame_image is not None:
                    frame_image_ip = ImageProcessor(frame_image)
                    frame_image_ip.median_blur(5, opacity=self.median_blur_per / 100.0 )
                    frame_image_ip.reresize( self.degrade_bicubic_per / 100.0, interpolation=ImageProcessor.Interpolation.CUBIC)
                    frame_image = frame_image_ip.get_image('HWC')
                return frame_image


import numexpr as ne
from xlib import avecl as lib_cl

class FaceMerger():

    def __init__(self):
        self.face_x_offset = 0.0
        self.face_y_offset = 0.0
        self.face_scale = 1.0
        self.face_mask_source = True
        self.face_mask_celeb = True
        self.face_mask_lmrks = False
        self.face_mask_erode = 5.0
        self.face_mask_blur = 5.0
        self.color_transfer = 'rct'
        self.interpolation = 'bilinear'
        self.color_compression = 10.0
        self.face_opacity = 1.0
        self.device = "CPU"

    _cpu_interp = {'bilinear' : ImageProcessor.Interpolation.LINEAR,
                   'bicubic'  : ImageProcessor.Interpolation.CUBIC,
                   'lanczos4' : ImageProcessor.Interpolation.LANCZOS4}

    def _merge_on_cpu(self, frame_image, face_resolution, face_align_img, face_align_mask_img, face_align_lmrks_mask_img, face_swap_img, face_swap_mask_img, aligned_to_source_uni_mat, frame_width, frame_height, do_color_compression ):


        interpolation = FaceMerger._cpu_interp[self.interpolation]

        frame_image = ImageProcessor(frame_image).to_ufloat32().get_image('HWC')

        masks = []
        if self.face_mask_source:
            masks.append( ImageProcessor(face_align_mask_img).to_ufloat32().get_image('HW') )
        if self.face_mask_celeb:
            masks.append( ImageProcessor(face_swap_mask_img).to_ufloat32().get_image('HW') )
        if self.face_mask_lmrks:
            masks.append( ImageProcessor(face_align_lmrks_mask_img).to_ufloat32().get_image('HW') )

        masks_count = len(masks)
        if masks_count == 0:
            face_mask = np.ones(shape=(face_resolution, face_resolution), dtype=np.float32)
        else:
            face_mask = masks[0]
            for i in range(1, masks_count):
                face_mask *= masks[i]

        # Combine face mask
        face_mask = ImageProcessor(face_mask).erode_blur(self.face_mask_erode, self.face_mask_blur, fade_to_border=True).get_image('HWC')
        frame_face_mask = ImageProcessor(face_mask).warp_affine(aligned_to_source_uni_mat, frame_width, frame_height).clip2( (1.0/255.0), 0.0, 1.0, 1.0).get_image('HWC')

        face_swap_ip = ImageProcessor(face_swap_img).to_ufloat32()

        if self.color_transfer == 'rct':            
            face_swap_img = face_swap_ip.rct(like=face_align_img, mask=face_mask, like_mask=face_mask)

        frame_face_swap_img = face_swap_ip.warp_affine(aligned_to_source_uni_mat, frame_width, frame_height, interpolation=interpolation).get_image('HWC')

        # Combine final frame
        opacity = np.float32(self.face_opacity)
        one_f = np.float32(1.0)
        if opacity == 1.0:
            out_merged_frame = ne.evaluate('frame_image*(one_f-frame_face_mask) + frame_face_swap_img*frame_face_mask')
        else:
            out_merged_frame = ne.evaluate('frame_image*(one_f-frame_face_mask) + frame_image*frame_face_mask*(one_f-opacity) + frame_face_swap_img*frame_face_mask*opacity')

        if do_color_compression and self.color_compression != 0:
            color_compression = max(4, (127.0 - self.color_compression) )
            out_merged_frame *= color_compression
            np.floor(out_merged_frame, out=out_merged_frame)
            out_merged_frame /= color_compression
            out_merged_frame += 2.0 / color_compression

        return out_merged_frame

    _gpu_interp = {'bilinear' : lib_cl.EInterpolation.LINEAR,
                   'bicubic'  : lib_cl.EInterpolation.CUBIC,
                   'lanczos4' : lib_cl.EInterpolation.LANCZOS4}

    _n_mask_multiply_op_text = [ f"float X = {'*'.join([f'(((float)I{i}) / 255.0)' for i in range(n)])}; O = (X <= 0.5 ? 0 : 1);" for n in range(5) ]

    def _merge_on_gpu(self, frame_image, face_resolution, face_align_img, face_align_mask_img, face_align_lmrks_mask_img, face_swap_img, face_swap_mask_img, aligned_to_source_uni_mat, frame_width, frame_height, do_color_compression ):
        self = self.get_self()
        interpolation = self._gpu_interp[self.interpolation]

        masks = []
        if self.face_mask_source:
            masks.append( lib_cl.Tensor.from_value(face_align_mask_img) )
        if self.face_mask_celeb:
            masks.append( lib_cl.Tensor.from_value(face_swap_mask_img) )
        if self.face_mask_lmrks:
            masks.append( lib_cl.Tensor.from_value(face_align_lmrks_mask_img) )

        masks_count = len(masks)
        if masks_count == 0:
            face_mask_t = lib_cl.Tensor(shape=(face_resolution, face_resolution), dtype=np.float32, initializer=lib_cl.InitConst(1.0))
        else:
            face_mask_t = lib_cl.any_wise(FaceMerger._n_mask_multiply_op_text[masks_count], *masks, dtype=np.uint8).transpose( (2,0,1) )

        face_mask_t = lib_cl.binary_morph(face_mask_t, self.face_mask_erode, self.face_mask_blur, fade_to_border=True, dtype=np.float32)
        face_swap_img_t  = lib_cl.Tensor.from_value(face_swap_img ).transpose( (2,0,1), op_text='O = ((O_TYPE)I) / 255.0', dtype=np.float32)

        if self.color_transfer == 'rct':
            face_align_img_t = lib_cl.Tensor.from_value(face_align_img).transpose( (2,0,1), op_text='O = ((O_TYPE)I) / 255.0', dtype=np.float32)
            face_swap_img_t = lib_cl.rct(face_swap_img_t, face_align_img_t, target_mask_t=face_mask_t, source_mask_t=face_mask_t)

        frame_face_mask_t     = lib_cl.remap_np_affine(face_mask_t,     aligned_to_source_uni_mat, interpolation=lib_cl.EInterpolation.LINEAR, output_size=(frame_height, frame_width), post_op_text='O = (O <= (1.0/255.0) ? 0.0 : O > 1.0 ? 1.0 : O);' )
        frame_face_swap_img_t = lib_cl.remap_np_affine(face_swap_img_t, aligned_to_source_uni_mat, interpolation=interpolation, output_size=(frame_height, frame_width), post_op_text='O = clamp(O, 0.0, 1.0);' )

        frame_image_t = lib_cl.Tensor.from_value(frame_image).transpose( (2,0,1), op_text='O = ((float)I) / 255.0;' if frame_image.dtype == np.uint8 else None,
                                                                                  dtype=np.float32 if frame_image.dtype == np.uint8 else None)

        opacity = self.face_opacity
        if opacity == 1.0:
            frame_final_t = lib_cl.any_wise('O = I0*(1.0-I1) + I2*I1', frame_image_t, frame_face_mask_t, frame_face_swap_img_t, dtype=np.float32)
        else:
            frame_final_t = lib_cl.any_wise('O = I0*(1.0-I1) + I0*I1*(1.0-I3) + I2*I1*I3', frame_image_t, frame_face_mask_t, frame_face_swap_img_t, np.float32(opacity), dtype=np.float32)

        if do_color_compression and self.color_compression != 0:
            color_compression = max(4, (127.0 - self.color_compression) )
            frame_final_t = lib_cl.any_wise('O = ( floor(I0 * I1) / I1 ) + (2.0 / I1);', frame_final_t, np.float32(color_compression))

        return frame_final_t.transpose( (1,2,0) ).np()



    def on_tick(self,merged_frame, fsi_list, face_align_img, face_align_lmrks_mask_img, face_align_mask_img, face_swap_img, face_swap_mask_img):



                if merged_frame is not None:
                    fsi_list_len = len(fsi_list)
                    has_merged_faces = False

                    for fsi_id, fsi in enumerate(fsi_list):



                        image_to_align_uni_mat = fsi.image_to_align_uni_mat
                        face_resolution        = fsi.face_resolution


                        if all_is_not_None(face_resolution, face_align_img, face_align_mask_img, face_swap_img, face_swap_mask_img, image_to_align_uni_mat):
                            has_merged_faces = True
                            face_height, face_width = face_align_img.shape[:2]
                            frame_height, frame_width = merged_frame.shape[:2]
                            aligned_to_source_uni_mat = image_to_align_uni_mat.invert()
                            aligned_to_source_uni_mat = aligned_to_source_uni_mat.source_translated(-self.face_x_offset, -self.face_y_offset)
                            aligned_to_source_uni_mat = aligned_to_source_uni_mat.source_scaled_around_center(self.face_scale,self.face_scale)
                            aligned_to_source_uni_mat = aligned_to_source_uni_mat.to_exact_mat (face_width, face_height, frame_width, frame_height)
                            do_color_compression = fsi_id == fsi_list_len-1
                            if self.device == 'CPU':
                                merged_frame = self._merge_on_cpu(merged_frame, face_resolution, face_align_img, face_align_mask_img, face_align_lmrks_mask_img, face_swap_img, face_swap_mask_img, aligned_to_source_uni_mat, frame_width, frame_height, do_color_compression=do_color_compression )
                            else:
                                merged_frame = self._merge_on_gpu(merged_frame, face_resolution, face_align_img, face_align_mask_img, face_align_lmrks_mask_img, face_swap_img, face_swap_mask_img, aligned_to_source_uni_mat, frame_width, frame_height, do_color_compression=do_color_compression )

                    if has_merged_faces:
                        # keep image in float32 in order not to extra load FaceMerger
                        # merged_image_name = f'{frame_image_name}_merged'
                        # bcd.set_merged_image_name(merged_image_name)
                        # bcd.set_image(merged_image_name, merged_frame)
                        return merged_frame














class DeepFaceLiveApp():
    def __init__(self,userdata_path) -> None:
         self.camera = Camera()
         self.faceDetector = FacDetetor()
         self.faceMarker = FaceMarker()
         self.faceAligner = FaceAligner()
         self.faceSwapper = FaceSwapper()
         self.frameAjuster = FrameAdjuster()
         self.faceMerger = FaceMerger()

    def run(self):

        while True:
            img = self.camera.on_tick()
            swap_info_list = self.faceDetector.on_tick(img)
            self.faceMarker.on_tick(img, swap_info_list)
            face_align_lmrks_mask_img , face_align_img = self.faceAligner.on_tick(img, swap_info_list)
            (face_align_mask_img, face_swap_img, face_swap_mask_img) = self.faceSwapper.on_tick(face_align_img, swap_info_list)
            frameAjuster_img = self.frameAjuster.on_tick(img)

            res_img = self.faceMerger.on_tick(frameAjuster_img, swap_info_list, face_align_img, face_align_lmrks_mask_img, face_align_mask_img, face_swap_img, face_swap_mask_img)

            if face_swap_img is not None:           
                cv2.imshow('client', res_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


    


