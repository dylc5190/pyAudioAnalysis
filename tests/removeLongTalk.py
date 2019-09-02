'''
python3 audioAnalysis.py trainClassifier -i data/100/train/speak/ data/100/train/sing/ --method svm -o data/svmSM
python3 ../tests/removeLongTalk.py -o=data/300/result.mp3 -i=data/300/wncw_20190825.mp3 --model svm --classifier data/svmSM

'''
from pydub import AudioSegment
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioFeatureExtraction as aF
import os
import sys
import argparse
import datetime

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", required=True, help="Input audio file")
    parser.add_argument("-o", "--output", help="Output audio file")
    parser.add_argument("--model", choices=["svm", "svm_rbf", "knn",
                                            "randomforest",
                                            "gradientboosting",
                                            "extratrees"],
                        required=True, help="Classifier type (svm or knn or"
                                            " randomforest or "
                                            "gradientboosting or "
                                            "extratrees)")
    parser.add_argument("--classifier", required=True,
                           help="Classifier to use (path)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    model_name = args.classifier
    model_type = args.model
    inputFile = args.input

    if not os.path.isfile(model_name):
        raise Exception("Input model_name not found!")
    if not os.path.isfile(inputFile):
        raise Exception("Input audio file not found!")

    if model_type == 'knn':
        [classifier, MEAN, STD, classNames, mt_win, mt_step, st_win, st_step,
         compute_beat] = load_model_knn(model_name)
    else:
        [classifier, MEAN, STD, classNames, mt_win, mt_step, st_win, st_step,
         compute_beat] = aT.load_model(model_name)

    audiofile = AudioSegment.from_mp3(inputFile)
    interval = 5
    slices = audiofile[::interval*1000]
    speak = 0
    music = 0
    buffers = []
    output = AudioSegment.empty()
    for j, slice in enumerate(slices):

        [Fs, x] = audioBasicIO.readAudioSegment(slice)
        x = audioBasicIO.stereo2mono(x)

        if isinstance(x, int):                                 # audio file IO problem
            print("Audio segment {} has problem".format(j))
            continue
        if x.shape[0] / float(Fs) <= mt_win:
            print("Audio segment {} has problem".format(j))
            continue

        # feature extraction:
        [mt_features, s, _] = aF.mtFeatureExtraction(x, Fs, mt_win * Fs, mt_step * Fs, round(Fs * st_win), round(Fs * st_step))
        mt_features = mt_features.mean(axis=1)        # long term averaging of mid-term statistics
        if compute_beat:
            [beat, beatConf] = aF.beatExtraction(s, st_step)
            mt_features = numpy.append(mt_features, beat)
            mt_features = numpy.append(mt_features, beatConf)
        curFV = (mt_features - MEAN) / STD                # normalization

        [Result, P] = aT.classifierWrapper(classifier, model_type, curFV)    # classification

        #print("[{:04d},{}]".format(j,Result))
        t = str(datetime.timedelta(seconds=j*interval))
        if Result == 0: #speak
           if speak == 0:
              print("[{:04d},{}] start of talk".format(j,t))
              speak = j
           buffers.append(slice)
           music = 0
        else:
           if speak:
              music += 1
              if music > 2:	# hyper-parameter
                 if j-speak > 150//interval:
                    print("[{:04d},{}] Possible news report or advertisement".format(j,t))
                    for chunk in buffers[-2:]: output += chunk
                    output += slice
                 else:
                    print("[{:04d},{}] DJ's talk".format(j,t))
                    for chunk in buffers: output += chunk
                 buffers = []
                 speak = 0
                 music = 0
              else:
                 print("[{:04d},{}] Music starts or possible FA".format(j,t))
                 #slice.export("{}-{}.mp3".format(j,t).replace(':','-'),format="mp3")
                 buffers.append(slice)
           else:
              output += slice

        #print("{0:s}\t{1:s}".format("Class", "Probability"))
        #for i, c in enumerate(classNames):
        #    print("{0:s}\t{1:.2f}".format(c, P[i]))
        #print("Winner class: " + classNames[int(Result)])

    if args.output: output.export(args.output,format="mp3")
