import os.path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)


def main():
    path = os.path.join(project_dir, 'data', 'sms_cls_train.csv')
    df = pd.read_csv(path, encoding='latin', sep='\t', header=None, names=['label', 'text'])
    df['type'] = df['label'].map(lambda a: 1 if a == 'ham' else 0)

    # tf-idf用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。
    # 使用tf-idf构建词向量，可以较好的提取出文本内容强相关的词汇，把一些没有实际的词语筛除掉
    tf_vect = TfidfVectorizer(binary=True)
    nb_model = MultinomialNB(alpha=1, fit_prior=True)

    # 创建pipe的模型，如果模型不包含前面的vectorizer，会导致在推理计算的时候找不到相应的词向量
    pipe_model = Pipeline([("vectorizer", tf_vect), ("classifier", nb_model)])

    # 训练集、测试集切分
    x_train, x_test, y_train, y_test = train_test_split(df.text, df.type, test_size=0.20, random_state=100)
    print("train count: ", x_train.shape[0], "test count: ", x_test.shape[0])

    # 训练模型
    pipe_model.fit(x_train, y_train)

    # 评估模型
    y_pred = pipe_model.predict(x_test)
    print("accuracy on test data: ", accuracy_score(y_test, y_pred))

    # 保存模型
    model_dir = os.path.join(project_dir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'naive_bayes.pkl')
    joblib.dump(pipe_model, model_path)

    print('finish train')


if __name__ == '__main__':
    main()
