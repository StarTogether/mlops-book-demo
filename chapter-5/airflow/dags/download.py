import os
import requests
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, required=False, default='train')


def main():
    args = parser.parse_args()
    data_dir = os.path.join(project_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if args.mode == 'train':
        data_url = 'https://github.com/StarTogether/mlops-book-demo/raw/master/chapter-5/data/sms_cls_train.csv'
        data_path = os.path.join(data_dir, "sms_cls_train.csv")
    else:
        data_url = 'https://github.com/StarTogether/mlops-book-demo/raw/master/chapter-5/data/sms_cls_inference.csv'
        data_path = os.path.join(data_dir, "sms_cls_inference.csv")

    response = requests.get(data_url)
    with open(data_path, "wb") as f:
        f.write(response.content)

    print('finish download')


if __name__ == '__main__':
    main()
