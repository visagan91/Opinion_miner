# Opinion_minerOpinion miner with a comparative experiment component

Introduction:
The objective of this assignment is to implement a opinion miner that operates on the given data while evaluating design of the very build. 
In the pervious assignment we have already analysed the dataset and strategised the system design. This report explains implementation of that system design while also acknowledging the subjected changes with respect to design and algorithmic decisions.
Implementation:
The system design is proposed with multiple sub-tasks that deals with specific processes pipelined for optimal efficiency. The system design as mentioned in assignment one(Figure 1) is followed. 
This documentation is written in the way specified in the assignment, thus one or multiple sub-tasks in the system design with the specific function is mentioned and explained accordingly. 
Product feature extraction:
On analysis it was evident that the data is diverse in multiple aspects. Thus making essential ingestion and preprocessing essential even before exploratory analysis for product feature extraction. 
Initial ingestion of data is designed to convert the .txt document in the given data into CSV files. As the data set is diverse with respect to rows and columns and also the vocabulary effective pre-processing techniques are used.
Parsing: All files with ‘.txt’ extension is converted into CSVs and store in separate files and also a combined CSV is generated.
Cleaning: Removal of lowercase, whitespace, punctuations.
Normalisation: Normalisation with respect to aspect synonyms is achieved. 
Linguistic feature extraction: PoS tagging using spaCy and noun phrases are identified.
These methods effectively paved way for the extraction of useful( Table 1) that are also further processed for as desired for next steps.
Effective mapping of variants of a feature into one single normalised feature is achieved.Few examples of a normalised final features to the raw feature present in the dataset is also shown.(Table 2)

Sentiment analysis :
Exploratory data analysis : Patterns in the aspect and sentiments are analysed and visualised for using in relational mapping.
The extracted features are mapped into plots using per determined plot libraries available for better understanding of the cleaned and normalised data.
Sentiment plots: Two bar chart that shows the overall count of positive and the negative features with respect to strength and distribution across all mentions features is used to identify the sentiment bias.
Sentence length plot: A histogram that analyses the number of words per sentence that helps to detect inconsistencies in the cleaned data.
Frequency of POS Tag / Noun phrase : Bar charts that lists the most frequently appearing POS tag and noun phrases are shown to help in key grammatical structures and aspect detection respectively.
Domain wise sentiment distribution plot : A stacked bar chart that compares the domain level distribution of sentiments(both positive and negative).
Sentiment vs Strength heat map: A heat map is used to show the count of sentiment to strength combinations. This particularly helps in understanding the distribution dynamics of sentiments. 
Now, when proceeding to the relation mapping stage in the proposed system design, the particular sub-task is identified to be optimal for the comparative experiment where two methods are used and compared among them on implementation as specified in the assignment. 
Hence, the relation mapping to link each product feature mentioned to the sentiment polarity expressed toward that feature is achieved by two approaches. 
Rule-Based relation mapping:
In the basic and simple approach we use linguistic rules and dependency parsing to achieve effective mapping of a sentiment into a extracted feature.
Parsing is done with spaCy and search for sentiment is specific to adjectives near that using modifier ADJ. Then checked for negative or positive lexicon before assigning accordingly.
BERT-Based relation mapping:
Uses a zero-shot classification pre-trained model from Hugging face that maps sentiment to a feature using NLI(Natural Language Interface).
A generated hypothesis is tested against sentiments(positive, neutral, negative) to test the score before mapping into the corresponding sentiment of the highest score.
Sentiment aggregation: Identifies features of the customers and user priorities to aggregate sentiments per product feature. 
A option to select from either of the relation mapping model to work on is given at this stage.

Evaluation and discussion:
The opinion miner that extract feature-level sentiments from reviews. It performs a aspect-based analysis on a multi-domain dataset.
Data Ingestion:
The proposed method of ingestion is well preserved and well-prepared for cross-domain comparisons. The dataset keeping track of the source folder by adding a domain column is crucial for domain-specific analysis.


Preprocessing:
Techniques used in cleaning and normalisation is effective and consistent. Rule-based mapping thrives on the linguistic normalisation achieved at this stage.
Exploratory data analysis:
Visualisation offered is great for interpretability and throws focus insights with respect to linguistic and opinion aspects.
Relation-mapping:
Since two parallel approaches is used, evaluation and comparing the two methods is used as operate element in code to analysis and visualise the differences between them.
Comparative experiment: On analysis on the outputs of both the methods we conclude the following results.
Though rule-based method is simple, transparent and requires minimal resources it depends heavily on the grammar of the reviews. While in Bert-based pre trained model is relatively close in identifying the slang grammar in the reviews.
Bert-based relation mapping is time consuming and sometimes ambiguous on mapping sentiment as depends on contextual cues.
Two plots are generated to show the agreement rate and confusion matrix with respect to the prediction of the two model. 
Opinion aggregation:
Operation on the user selective relation mapping model is easy to operate of varying use preferences. 

Conclusion:
The proposed opinion miner system is well-pipelined combines techniques of optimal data-processing and analysis that allows for both human understandable and model-driven interpretations.
