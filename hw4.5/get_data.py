"""Fetch the HW4.a Twitter dataset (Twitter US Airline Sentiment, Kaggle crowdflower/twitter-airline-sentiment)
to data/Tweets.csv and print its sha256 so it can be checked against the HW4.a data lock.

    python get_data.py                     # kagglehub download (public, no API key needed)
    python get_data.py --src /kaggle/input/twitter-airline-sentiment/Tweets.csv   # attached Kaggle input / own copy
"""
import argparse
import glob
import os
import shutil

from bert_common import sha256_file

DST = "data/Tweets.csv"
# sha256 of Tweets.csv from version 4 of the Kaggle dataset
KNOWN_SHA256 = "ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="", help="existing Tweets.csv to copy instead of downloading")
    a = ap.parse_args()
    src = a.src
    if not src:
        attached = glob.glob("/kaggle/input/**/Tweets.csv", recursive=True)
        if attached:
            src = attached[0]
        else:
            import kagglehub
            src = os.path.join(kagglehub.dataset_download("crowdflower/twitter-airline-sentiment"), "Tweets.csv")
    os.makedirs("data", exist_ok=True)
    shutil.copyfile(src, DST)
    h = sha256_file(DST)
    print(f"{src} -> {DST}\nsha256 {h} ({'matches' if h == KNOWN_SHA256 else 'DIFFERS FROM'} the Kaggle v4 file)")


if __name__ == "__main__":
    main()
