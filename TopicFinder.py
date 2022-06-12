import attr
from keybert import KeyBERT
from utils import load_zeroshot_model

import nltk

# nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import wordnet as wn

classifier = load_zeroshot_model()

@attr.s
class TopicFinder:
    """
    Identifies topics in the given piece of text using wordnet and zero-shot classifiers.
    step 1: Use transformer model to find the keywords form the text
    step 2: Use the wordnet to find most similar words and expand the keyword list.
    step 3: Find the hypernyms for all the keywords.
    step 4: Pass the list of hypernyms and text to your choice of zero-shot classifier and get the top labels as topic.
    """

    model = attr.ib(default='all-MiniLM-L6-v2')

    def __attrs_post_init__(self):
        self.model = KeyBERT(self.model)

    def find_topic(self, text, n_topic=2):
        """
        Infer the topic in a given string.
        parameters
        ----------
        text: str
            pass the text for which you want to infer the topic.
        n_topic: int
            Define the maximum number of topic you want to identify.
        Returns
        -------
        labels : list
            List of topics identified.
        """
        keyword = self.get_keyword(text)
        labels = self.get_parent_words(keyword)
        prediction = classifier(text, candidate_labels=labels)
        labels = prediction['labels'][:n_topic]
        labels = [i.replace("_", ' ').title() for i in labels]
        return labels

    def get_keyword(self, text):
        kw = [i[0] for i in self.model.extract_keywords(text)]
        return kw

    def get_parent_words(self, keywords):
        parents = []
        for kw in keywords:
            sym = wn.synsets(kw)[:2]
            parents += [j.name().split('.')
                        for i in sym for j in i.hypernyms()]
        parents = [i[0] for i in parents if i[1] != 'v']
        parents = [i for i in parents if i not in keywords]
        parents = list(set(parents))
        return parents


def main():
    model = TopicFinder()
    txt = """Dinosaurs are a varied group of animals from taxonomic, morphological and ecological standpoints. Birds, at over 10,700 living species, 
are among the most diverse group of vertebrates. Using fossil evidence, paleontologists have identified over 900 distinct genera and more than
1,000 different species of non-avian dinosaurs. Dinosaurs are represented on every continent by both extant species (birds) and fossil remains.
Through the first half of the 20th century, before birds were recognized as dinosaurs, most of the scientific community believed dinosaurs to have
been sluggish and cold-blooded. Most research conducted since the 1970s, however, has indicated that dinosaurs were active animals with elevated
metabolisms and numerous adaptations for social interaction. Some were herbivorous, others carnivorous. 
Evidence suggests that all dinosaurs were egg-laying; and that nest-building was a trait shared by many dinosaurs, both avian and non-avian. """


    print(model.find_topic(txt, n_topic=1))


main()
