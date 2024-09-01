from src.pykeen.pipeline import pipeline

if __name__ == '__main__':

    result = pipeline(
        model='TransE',
        dataset='FB15k',
    )