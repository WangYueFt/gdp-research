"""Tool to convert Waymo Open Dataset to tf.Examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import apache_beam as beam
from pillar.data import waymo_decoderfrom waymo_open_dataset import dataset_pb2

tf.enable_v2_behavior()

flags.DEFINE_string('input_file_pattern', None, 'Path to read input')
flags.DEFINE_string('output_filebase', None, 'Path to write output')

FLAGS = flags.FLAGS


def main(unused_argv):

  assert FLAGS.input_file_pattern
  assert FLAGS.output_filebase

  reader = beam.io.tfrecordio.ReadFromTFRecord(
      FLAGS.input_file_pattern,
      coder=beam.coders.ProtoCoder(dataset_pb2.Frame))

  writer = beam.io.tfrecordio.WriteToTFRecord(
      FLAGS.output_filebase,
      coder=beam.coders.BytesCoder())

  with beam.Pipeline() as root:
    (root  # pylint: disable=expression-not-assigned
     | 'Read Frame TFRecords' >> reader
     | 'Reshuffle' >> beam.Reshuffle()
     | 'Decode Frame Proto' >> beam.Map(waymo_decoder.decode_frame)
     | 'Write SSTables' >> writer)


if __name__ == '__main__':
  app.run(main)
