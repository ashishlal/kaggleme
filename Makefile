data/sample_submission.csv.7z:
	wget -x --load-cookies data/cookies.txt https://www.kaggle.com/c/mercari-price-suggestion-challenge/download/sample_submission.csv.7z -O data/sample_submission.csv.7z --continue

data/test.tsv.7z:
	wget -x --load-cookies data/cookies.txt https://www.kaggle.com/c/mercari-price-suggestion-challenge/download/test.tsv.7z -O data/test.tsv.7z --continue

data/train.tsv.7z:
	wget -x --load-cookies data/cookies.txt https://www.kaggle.com/c/mercari-price-suggestion-challenge/download/train.tsv.7z -O data/train.tsv.7z --continue

data: data/sample_submission.csv.7z data/test.tsv.7z data/train.tsv.7z

clean: 
	rm -rf data/*.7z

.PHONY: data

