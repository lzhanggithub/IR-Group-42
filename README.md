# IR-Group-42
a repository that contains the software we used, the scripts we used to analyze our data, etc.
## Learning to Rank instructions
Type this into the command-line at location <code>...\IR-Group-42\Learning to Rank</code>: 

<code>java -jar bin/RankLib.jar -train validation/feature_train.txt -validate validation/feature_valid.txt -test validation/feature_test.txt -ranker 7 -metric2t NDCG@10 -save mymodel.txt </code>.

-ranker 7 specified means we want to train a ListNet ranker: train on the training data and record the model that performs best on the validation data. The training metric is NDCG@10. Finally, the model will be saved to a file named mymodel.txt in the current directory.

More details such as changing models, select which features to use in the algorithm and other evaluation matrix are [RankLib Overview](https://sourceforge.net/p/lemur/wiki/RankLib/).
