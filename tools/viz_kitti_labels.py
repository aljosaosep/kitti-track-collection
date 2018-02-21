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

import os
import argparse

from PIL import Image, ImageDraw, ImageFont


class TrackData:
	def __init__(self):
		self.frame = None
		self.track_id = None
		self.obj_str = None
		self.x1 = None
		self.y1 = None
		self.x2 = None
		self.y2 = None


tiny_little_palette = ['#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd',\
					'#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22',\
					'#dbdb8d', '#17becf', '#9edae5', '#1f77b4']*5


def main(_):

	# Get font handle (needed for viz)
	font_path = FLAGS.font_path
	if not os.path.isfile(font_path):
		raise Exception("Could not find font: %s."%font_path)
	font = ImageFont.truetype(font_path, 16, encoding="unic")

	# Make sure tracking data specified exists
	if not os.path.isfile(FLAGS.tracker_path):
		raise Exception("Could not find tracks: %s."%FLAGS.tracker_path)
	file_dir, file_name = os.path.split(FLAGS.tracker_path)
	seq_name = file_name[0:file_name.find(".txt")]
	this_seq_image_path = FLAGS.image_data_path.replace("%SEQUENCE%", seq_name)

	# Make sure that the output_dir is there
	if not os.path.isdir(FLAGS.output_dir):
		raise Exception("Output dir %s does not exist."%FLAGS.output_dir)

	# -----------------------------------
	# Parse data for the whole sequence
	# -----------------------------------
	f_tracker = open(FLAGS.tracker_path, "r")
	track_data = []
	for line in f_tracker:
		line = line.strip()
		fields = line.split(" ")
		tdata = TrackData()
		tdata.frame = int(float(fields[0]))
		tdata.track_id = int(float(fields[1]))
		tdata.obj_str = fields[2].lower()
		tdata.x1 = float(fields[6])
		tdata.y1 = float(fields[7])
		tdata.x2 = float(fields[8])
		tdata.y2 = float(fields[9])
		track_data.append(tdata)
	f_tracker.close()

	# -----------------------------------
	# Visualize data
	# -----------------------------------
	# Organize data into maps: { (frame, [obj_list]), ... }
	tracking_data_map = {}
	for tdata in track_data:
		frame = tdata.frame

		# No frame entry yet -> add one
		if frame not in tracking_data_map:
			tracking_data_map[frame] = []

		# Insert this tdata
		tracking_data_map[frame].append(tdata)

	# Now visualize this stuff
	for frame in tracking_data_map.keys():
		print ('Viz frame %d ...'%frame)

		# Load the image
		image_path = this_seq_image_path%frame

		if not os.path.isfile(image_path):
			raise Exception("Could not find image: %s."%image_path)

		im = Image.open(image_path)
		draw = ImageDraw.Draw(im)

		# Get all obj for this frame and go through
		objs = tracking_data_map[frame]
		for obj in objs:
			color = tiny_little_palette[obj.track_id] # Get color based on the track id (lookup table)

			# Draw rect + text
			draw.rectangle(((obj.x1, obj.y1), (obj.x2, obj.y2)), outline=color)
			draw.text((obj.x1, obj.y1), obj.obj_str, color, font=font)
			draw.text((obj.x1, obj.y1+15), str(obj.track_id), color, font=font)

		# Done, save the output image
		head, tail = os.path.split(image_path)
		im.save(os.path.join(FLAGS.output_dir, tail))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'tracker_path',
		type=str,
		help='Specify file containing tracks, eg. /home/dog/projects/kitti-track-collection/tracks/kitti_format/2011_09_26_drive_0001.txt'
	)

	parser.add_argument(
		'image_data_path',
		type=str,
		help='Specify path to images, eg. /home/dog/datasets/kitti_raw_preprocessed/%SEQUENCE%/image_02/%010d.png'
	)

	parser.add_argument(
		'output_dir',
		type=str,
		help='Eg. /tmp/output_dir'
	)

	parser.add_argument(
		'--font_path',
		type=str,
		default='/usr/share/fonts/truetype/freefont/FreeMono.ttf',
		help='Specify font path you want to use (optional)'
	)

	FLAGS = parser.parse_args()
	main(FLAGS)
