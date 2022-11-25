import os
import pandas as pd
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)


def main():
    model_path = os.path.join(project_dir, 'model', 'naive_bayes.pkl')
    data_path = os.path.join(project_dir, 'data', 'sms_cls_inference.csv')
    result_path = os.path.join(project_dir, 'data', 'sms_cls_inference_result.csv')

    pipe_model = joblib.load(model_path)

    df = pd.read_csv(data_path, encoding='latin', sep='\t', header=None, names=['text', 'type', 'label'])

    result = pipe_model.predict(df.text)
    df['type'] = result
    df['label'] = df['type'].map(lambda a: 'spam' if a == 1 else 'ham')

    # 保存推理结果，实际业务中可以上传到hive等数据仓库中
    df.to_csv(result_path, columns=['label', 'text'], index=False)

    print('finish inference')


if __name__ == '__main__':
    main()
