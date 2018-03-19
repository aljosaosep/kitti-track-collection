"""
Copyright 2017. All rights reserved.
Computer Vision Group, Visual Computing Institute
RWTH Aachen University, Germany

This file is part of the Kitti Track Collection (KTC) devkit .
Authors:
Aljosa Osep (osep -at- vision.rwth-aachen.de),
Paul Voigtlaender (voigtlaender -at- vision.rwth-aachen.de),
Jonathon Luiten (jonathon.luiten -at- rwth-aachen.de),
Stefan Breuers (breuers -at- vision.rwth-aachen.de)

KTC devkit is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or any later version.

KTC devkit is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
KTC devkit; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA
"""

import os, glob, argparse
import hypotheses_pb2
import numpy as np
import pycocotools.mask as cocomask
import struct

from PIL import Image, ImageDraw

def MaskObjToRle(mask_obj):
	# Fill RLE dict
	rle = {}
	rle['counts'] = mask_obj.rle_string
	rle['size'] = [mask_obj.mask_height, mask_obj.mask_width]
	return rle

def UnpackMaskRLE(mask_obj):
	rle = MaskObjToRle(mask_obj)

	# Decode
	M = np.squeeze(cocomask.decode(rle))  # Decode counts->mask
	M_inds = np.flatnonzero(M > 0)  # Mask->inds, for those who like inds
	return (M, M_inds)

def get_frame_range(hypotheses):
	all_frames = []
	for hypo in hypotheses:
		all_frames.extend(hypo.timestamps)
	first_frame = min(all_frames)
	end_frame = max(all_frames)
	return (first_frame, end_frame)

def RectUnion(rect_1, rect_2):
	left = min([rect_1[0], rect_2[0]])
	right = max([rect_1[0] + rect_1[2], rect_2[0] + rect_2[2]])
	top = min([rect_1[1], rect_2[1]])
	bottom = max([rect_1[1] + rect_1[3], rect_2[1] + rect_2[3]])
	return [left, top, right - left, bottom - top]

def load_hypos_from_proto(protos_dir):
	if not os.path.isdir(protos_dir):
		raise Exception("Error, protos dir %s not found!" % protos_dir)

	proto_files = glob.glob(os.path.join(protos_dir, "*.txt"))
	if len(proto_files) <= 0:
		raise Exception("Error, to proto files in the proto dir %s.!" % protos_dir)

	print ("Found %d protos, yay!" % len(proto_files))

	all_hypos = {}
	for proto_path in proto_files:

		# Get file, sequence name
		proto_dir, proto_filename = os.path.split(proto_path)
		seq_name = proto_filename[0:proto_filename.find("_merged")]  # Sequence name goes until "_merged_..." substring
		hypothesis_set = hypotheses_pb2.HypothesisSet()
		try:
			f = open(proto_path, "rb")
			hypothesis_set.ParseFromString(f.read())
			f.close()
		except IOError:
			raise Exception('Could not open file: %s' % proto_path)

		all_hypos[seq_name] = hypothesis_set

	return all_hypos

not_so_tiny_palette = ['#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd',\
					'#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22',\
					'#dbdb8d', '#17becf', '#9edae5', '#1f77b4']*1000

def main(_):

	# ------------------------------------
	#   +++ Load all protos +++
	# ------------------------------------
	protos_dir = os.path.abspath("../tracks/protobuf_format")
	all_hypos = load_hypos_from_proto(protos_dir=protos_dir)

	# ------------------------------------
	#   +++ Transform to KITTI format +++
	# ------------------------------------
	output_dir = os.path.abspath(FLAGS.output_dir)

	if not os.path.isdir(output_dir):
		raise Exception("Error, no such output dir %s!" % output_dir)

	for seq_name in all_hypos.keys():
		print ("Processing KITTI sequence %s ..." % (seq_name))

		this_seq_image_path = FLAGS.image_data_path.replace("%SEQUENCE%", seq_name)

		hypos = all_hypos[seq_name].hypotheses
		first_frame, end_frame = get_frame_range(hypos)

		path_out = os.path.join(output_dir, '%s.txt'%seq_name)
		f_kitti = open(path_out, 'w')

		for frame in range(first_frame, end_frame + 1):


			# Fetch info from all hypos, defn. in this frame
			for (hypo_idx, hypo) in enumerate(hypos):

				if frame not in hypo.bounding_boxes_2D_with_timestamps.keys():
					continue

				# Load the image
				image_path = this_seq_image_path % frame

				if not os.path.isfile(image_path):
					raise Exception("Could not find image: %s." % image_path)

				im = Image.open(image_path)
				im = im.convert('RGBA')  # Add alpha, make things look sexy
				draw = ImageDraw.Draw(im)

				rect_union = None
				points_bg = []
				points = []
				points_first_frame = []
				for inlier_frame in hypo.bounding_boxes_2D_with_timestamps.keys():

					# Bounding box
					hypo_bbox = hypo.bounding_boxes_2D_with_timestamps[inlier_frame]

					# Track ID
					hypo_id = hypo.id

					# Category (labeled)
					hypo_annot_categ = hypo.annotated_category.name

					# 3D pos
					hypo_pos = hypo.poses_3D_camera_space_with_timestamps[inlier_frame]

					# Draw all masks
					color = not_so_tiny_palette[hypo_id]  # Get color based on the track id (lookup table)

					# Draw rect + text
					x1 = hypo_bbox.x0
					y1 = hypo_bbox.y0
					x2 = hypo_bbox.x0 + hypo_bbox.w
					y2 = hypo_bbox.y0 + hypo_bbox.h

					w,h = im.size
					if x1 < 0: x1 = 0
					if x2 > w: x2 = w-1
					if y1 < 0: y1 = 0
					if y2 > h: y2 = h-1

					rect_ = [x1, y1, x2-x1, y2-y1]
					if rect_union is None:
						rect_union = rect_
					else:
						rect_union = RectUnion(rect_union, rect_)

					# Get mask
					mask = hypo.masks_with_timestamps[inlier_frame]
					mask_image, mask_inds = UnpackMaskRLE(mask)
					nonzero_inds_x = np.where(mask_image == 1)[0]
					nonzero_inds_y = np.where(mask_image == 1)[1]
					for idx in range(0, len(nonzero_inds_x)):
						y = nonzero_inds_y[idx]
						x = nonzero_inds_x[idx]
						points.append((y, x))

						if inlier_frame == frame:
							points_first_frame.append((y, x))


					zero_inds_x = np.where(mask_image == 0)[0]
					zero_inds_y = np.where(mask_image == 0)[1]
					for idx in range(0, len(zero_inds_x)):
						y = zero_inds_y[idx]
						x = zero_inds_x[idx]
						points_bg.append((y, x))

				alpha = 0.5

				pts_bg_minus_fg = list(set(points_bg)-set(points))# [x for item in points_bg if item not in points]
				col_w = (255, 255, 255, int(255*alpha))
				draw.point(pts_bg_minus_fg, fill=col_w)

				color_tuple = struct.unpack('BBB', color[1::].decode('hex'))
				col_rgba = (color_tuple[0], color_tuple[1], color_tuple[2], int(255.0 * alpha))
				points_fg_minus_frame = list(set(points)-set(points_first_frame))# [x for item in points_bg if item not in points]
				draw.point(points_fg_minus_frame, fill=col_rgba)

				# Write image to dest dir

				# Crop img using rect_union
				width, height = im.size

				# Draw rect + text
				x1 = rect_union[0]
				y1 = rect_union[1]
				x2 = rect_union[0] + rect_union[2]
				y2 = rect_union[1] + rect_union[3]

				if x1 < 0: x1 = 0
				if x2 > width: x2 = width - 1
				if y1 < 0: y1 = 0
				if y2 > height: y2 = height - 1

				im2 = im.crop((x1, y1, x2, y2))  # (max(0, rect_union[0]), max(0,rect_union[1]), min(rect_union[2], width-1), min(rect_union[3], height-1)))

				im2.save(os.path.join(FLAGS.output_dir, 'seq_%s_hypo_%010d_frame_%010d.png'%(seq_name, hypo_idx, frame)))
				del draw

		f_kitti.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"image_data_path",
		type=str,
		help="Specify path to images, eg. /home/dog/datasets/kitti_raw_preprocessed/%SEQUENCE%/image_02/%010d.png."
	)

	parser.add_argument(
		"--output_dir",
		type=str,
		default="/tmp/viz_tracks",
		help="Specify track images output dir."
	)

	FLAGS = parser.parse_args()
	main(FLAGS)
