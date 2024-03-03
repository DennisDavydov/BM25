import re
import sys
#import readline  # NOQA
import math


class InvertedIndex:
    """
    A simple inverted index that uses BM25 scores.
    """

    def __init__(self):
        """
        Creates an empty inverted index.
        """

        self.inverted_lists = {}  # The inverted lists of doc ids.
        self.docs = []  # The docs, each in form (title, description).
        self.avdoclen = 0

    def build_from_file(self, file_name, b=0.75, k=1.75):
        """
        Construct the inverted index from the given file. The expected format
        of the file is one document per line, in the format
        <title>TAB<description>TAB<num_ratings>TAB<rating>TAB<num_sitelinks>
        Each entry in the inverted list associated to a word should contain a
        document id and a BM25 score. Compute the BM25 scores as follows:

        (1) In a first pass, compute the inverted lists with tf scores (that
            is the number of occurrences of the word within the <title> and the
            <description> of a document). Further, compute the document length
            (DL) for each document (that is the number of words in the <title>
            and the <description> of a document). Afterwards, compute the
            average document length (AVDL).
        (2) In a second pass, iterate over all inverted lists and replace the
            tf scores by BM25 scores, defined as:
            BM25 = tf * (k+1) / (k * (1 - b + b * DL/AVDL) + tf) * log2(N/df),
            where N is the total number of documents and df is the number of
            documents that contain the word.

        >>> ii = InvertedIndex()
        >>> ii.build_from_file("example.tsv", b=0, k=float("inf"))
        >>> inv_lists = sorted(ii.inverted_lists.items())
        >>> [(w, [(i, '%.3f' % tf) for i, tf in l]) for w, l in inv_lists]
        ... # doctest: +NORMALIZE_WHITESPACE
        [('animated', [(1, '0.415'), (2, '0.415'), (4, '0.415')]),
         ('animation', [(3, '2.000')]),
         ('film', [(2, '1.000'), (4, '1.000')]),
         ('movie', [(1, '0.000'), (2, '0.000'), (3, '0.000'), (4, '0.000')]),
         ('non', [(2, '2.000')]),
         ('short', [(3, '1.000'), (4, '2.000')])]

        >>> ii = InvertedIndex()
        >>> ii.build_from_file("example.tsv", b=0.75, k=1.75)
        >>> inv_lists = sorted(ii.inverted_lists.items())
        >>> [(w, [(i, '%.3f' % tf) for i, tf in l]) for w, l in inv_lists]
        ... # doctest: +NORMALIZE_WHITESPACE
        [('animated', [(1, '0.459'), (2, '0.402'), (4, '0.358')]),
         ('animation', [(3, '2.211')]),
         ('film', [(2, '0.969'), (4, '0.863')]),
         ('movie', [(1, '0.000'), (2, '0.000'), (3, '0.000'), (4, '0.000')]),
         ('non', [(2, '1.938')]),
         ('short', [(3, '1.106'), (4, '1.313')])]
        """

        with open(file_name, "r", encoding="utf8") as file:
            doc_id = 0
            for line in file:
                line = line.strip()

                doc_id += 1

                # Store the doc as a tuple (title, description).
                title, description, _ = line.split("\t", 2)
                self.docs.append((title, description, len (re.split("[^A-Za-z]+", line))))

                for word in re.split("[^A-Za-z]+", line):
                    word = word.lower().strip()

                    # Ignore the word if it is empty.
                    if len(word) == 0:
                        continue

                    if word not in self.inverted_lists:
                        # The word is seen for first time, create a new list.
                        self.inverted_lists[word] = [[doc_id, 1]]
                    elif self.inverted_lists[word][-1][0] != doc_id:
                        # Make sure that the list contains the id at most once.
                        self.inverted_lists[word].append([doc_id, 1])
                    elif self.inverted_lists[word][-1][0] == doc_id:
                        self.inverted_lists[word][-1][1] += 1
        
        
        for doc in self.docs:
            self.avdoclen = self.avdoclen + doc[2]
            
        self.avdoclen = self.avdoclen/len(self.docs)    
            
        for word in self.inverted_lists:
            self.bm25tf(k, b, word)
            self.bm25(k, b, word)
            
                
                
        # TODO: add your code to compute BM25 scores
    def bm25tf(self, k1, b, term):
        for doc in self.inverted_lists[term]:
            
            doc_id = doc[0] - 1
            #print(term)
            #print(doc)
            tf = b * self.docs[doc_id][2] / self.avdoclen
            tf = 1 - b + tf
            tf = k1 * tf
            tf = tf + doc[1]
            tf = doc[1] * (k1 + 1) / tf
            doc[1] = tf
    
    
    def bm25(self, k1, b, term):
        N = len(self.docs)
        n = len(self.inverted_lists[term])
        IDF = math.log2((N-n+0.5)/(n+0.5) + 1)
        
        for doc in self.inverted_lists[term]:
            doc[1] = doc[1]*IDF
            
        
        
    def merge(self, list1, list2):
        """
        Compute the union of the two given inverted lists in linear time
        (linear in the total number of entries in the two lists), where the
        entries in the inverted lists are postings of form (doc_id, bm25_score)
        and are expected to be sorted by doc_id, in ascending order.

        >>> ii = InvertedIndex()
        >>> l1 = ii.merge([(1, 2.1), (5, 3.2)], [(1, 1.7), (2, 1.3), (6, 3.3)])
        >>> [(id, "%.1f" % tf) for id, tf in l1]
        [(1, '3.8'), (2, '1.3'), (5, '3.2'), (6, '3.3')]

        >>> l2 = ii.merge([(3, 1.7), (5, 3.2), (7, 4.1)], [(1, 2.3), (5, 1.3)])
        >>> [(id, "%.1f" % tf) for id, tf in l2]
        [(1, '2.3'), (3, '1.7'), (5, '4.5'), (7, '4.1')]

        >>> l2 = ii.merge([], [(1, 2.3), (5, 1.3)])
        >>> [(id, "%.1f" % tf) for id, tf in l2]
        [(1, '2.3'), (5, '1.3')]

        >>> l2 = ii.merge([(1, 2.3)], [])
        >>> [(id, "%.1f" % tf) for id, tf in l2]
        [(1, '2.3')]

        >>> l2 = ii.merge([], [])
        >>> [(id, "%.1f" % tf) for id, tf in l2]
        []
        """
        
        union = []
        cursor1 = 0
        cursor2 = 0
        endpoint1 = len(list1)
        endpoint2 = len(list2)

        while cursor2 < endpoint2 and cursor1 < endpoint1:
            while list2[cursor2][0] < list1[cursor1][0]:
                union.append(list2[cursor2])
                cursor2 += 1
                #print(cursor2, 2)
                if cursor2>= endpoint2:
                    return union
            if list2[cursor2][0] == list1[cursor1][0]:
                union.append([list2[cursor2][0], list1[cursor1][1]+list2[cursor2][1]])
                #print(intersection)
                
            if list2[cursor2][0] > list1[cursor1][0]:
                union.append(list1[cursor1])
                #print(intersection)    
            cursor1 += 1
            
            #print(cursor1, 1)
        return union
        
        
    def process_query(self, query, use_refinements=False):
        """
        Process the given keyword query as follows: fetch the inverted list for
        each of the keywords in the query and compute the union of all lists.
        Sort the resulting list by BM25 scores in descending order.

        This method returns _all_ results for the given query, not just the
        top 3!

        If you want to implement some ranking refinements, make these
        refinements optional (their use should be controllable via the
        use_refinements flag).

        >>> ii = InvertedIndex()
        >>> ii.inverted_lists = {
        ... "foo": [(1, 0.2), (3, 0.6)],
        ... "bar": [(1, 0.4), (2, 0.7), (3, 0.5)],
        ... "baz": [(2, 0.1)]}
        >>> result = ii.process_query(["foo", "bar"], use_refinements=False)
        >>> [(id, "%.1f" % tf) for id, tf in result]
        [(3, '1.1'), (2, '0.7'), (1, '0.6')]
        """
        keywords = re.split("[^A-Za-z]+", query)
        if keywords[0] in self.inverted_lists:
            union = self.inverted_lists[keywords[0]]
        else:
            union = []
        for key in keywords:
            if  key in self.inverted_lists:
                union = self.merge(union, self.inverted_lists[key])
            else:
                union = []
                
        sorted_union = sorted(union, key=lambda x: x[1], reverse=True)
        
        sorted_pages = []
        for page in sorted_union:
            sorted_pages.append(page[0])
        return sorted_pages

        
def main():
    """
    Construct an inverted index from a given text file, then ask the user in
    an infinite loop for keyword queries and output the title and description
    of up to three matching records.
    """
 # Parse command line arguments and print usage information if needed
    if len(sys.argv) != 2:
        print("Usage: python3 ex.py <file name>")
        exit(1)

    file_name = sys.argv[1]

    # Create an inverted index from the given file
    ii = InvertedIndex()
    ii.build_from_file(file_name)
    
    while True:
        input1 = input("Input keywords to search for:")
        union = ii.process_query(input1)
        if len(union) != 0:
            for page in union[:3]:
                print("Title: " + ii.docs[page - 1][0])
            
                print("Description: "+ii.docs[page - 1][1] + '\n')   
        else: print("No matches found")           

if __name__ == "__main__":
    main()