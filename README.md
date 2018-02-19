# KITTI Track Collection (KTC) devkit

This repository contains the mined tracks and tools, extracted from KITTI Raw dataset in the scope of the following publication:
**Large-Scale Object Discovery and Detector Adaptation from Unlabeled Video [paper link](https://arxiv.org/pdf/1712.08832.pdf)**

By [Aljosa Osep](https://www.vision.rwth-aachen.de/person/13/), []Paul Voigtlaender](https://www.vision.rwth-aachen.de/person/197/) (equal contribution), Jonathon Luiten, Stefan Breuers, and Bastian Leibe

![Alt text](img/header.png?raw=true "KTC tracks.")

## Prerequisite
In order to use the python tools, your setup has to meet the following minimum requirements (tested versions in parentheses. Other versions might work, too):

* Python (2.7)
  * PIL or Pillow (3.4.2)
  * Google protobuf (3.4.x)
  
Note, that in order to access the image data, you need KITTI Raw dataset, you can get it [here](http://www.cvlibs.net/datasets/kitti/raw_data.php). 
We recommend using the [download script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip).

## Using the tracks data
You can find the mined tracks in `ROOT/tracks/kitti_format/%SEQUENCE_NAME%.txt` (KITTI format). For details about the data format, 
please see [KITTI tracking dataset web page](http://www.cvlibs.net/datasets/kitti/eval_tracking.php).

In addition, we also provide proto files, containing additional information (such as pixel masks), `ROOT/tracks/proto_format/%SEQUENCE_NAME%_plus_some_weird_stuff.txt`. 
The protbuf message format is defined in `SRC/proto/hypotheses.proto`.

## Using the tools
TODO

If you have any issues or questions with the data or the code, please contact me https://www.vision.rwth-aachen.de/person/13/

## Citing

If you find this data or tools useful in your reasearch, you should cite:

    @article{OsepVoigtlaender18arxiv,
        title={Large-Scale Object Discovery and Detector Adaptation from Unlabeled Video},
        author={Aljo\v{s}a O\v{s}ep and  Paul Voigtlaender and Jonathon Luiten and Stefan Breuers and Bastian Leibe},
        journal={arXiv preprint arXiv:1712.08832},
        year={2018}
    }

## License

GNU General Public License (http://www.gnu.org/licenses/gpl.html)

Copyright (c) 2018 Aljosa Osep, Paul Voigtlaender, Jonathon Luiten, Stefan Breuers
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.