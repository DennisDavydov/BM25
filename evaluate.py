
import sys
import re
import numpy as np


from inverted_index import InvertedIndex  # NOQA


class Evaluate:
    """
    Class for evaluating the InvertedIndex class against a benchmark.
    """
    def read_benchmark(self, file_name):
        """
        Read a benchmark from the given file. The expected format of the file
        is one query per line, with the ids of all documents relevant for that
        query, like: <query>TAB<id1>WHITESPACE<id2>WHITESPACE<id3> ...

        >>> evaluate = Evaluate()
        >>> benchmark = evaluate.read_benchmark("example-benchmark.tsv")
        >>> sorted(benchmark.items())
        [('animated film', {1, 3, 4}), ('short film', {3, 4})]
        """
        benchmarks = []
        with open(file_name, "r", encoding = "utf8") as file:
            for line in file:
                line = line.strip()
                query, documents = line.split("\t", 1)
                benchmarks.append((query, [int(x) for x in documents.split()]))
                    
        return benchmarks
                

    def evaluate(self, ii, benchmark, use_refinements=False, verbose=True):
        """
        Evaluate the given inverted index against the given benchmark as
        follows. Process each query in the benchmark with the given inverted
        index and compare the result list with the groundtruth in the
        benchmark. For each query, compute the measure P@3, P@R and AP as
        explained in the lecture. Aggregate the values to the three mean
        measures MP@3, MP@R and MAP and return them.

        Implement a parameter 'use_refinements' that controls the use of
        ranking refinements on calling the method process_query of your
        inverted index.

        >>> ii = InvertedIndex()
        >>> ii.build_from_file("example.tsv", b=0.75, k=1.75)
        >>> evaluator = Evaluate()
        >>> benchmark = evaluator.read_benchmark("example-benchmark.tsv")
        >>> measures = evaluator.evaluate(ii, benchmark, use_refinements=False,
        ... verbose=False)
        >>> [round(x, 3) for x in measures]
        [0.667, 0.833, 0.694]
        """
        MP3 = []
        MPR = []
        MAP = []
        
        for query, relevant_ids in benchmark:
            result_ids = ii.process_query(query, use_refinements)
            MP3.append(self.precision_at_k(result_ids, relevant_ids, 3))
            MPR.append(self.precision_at_k(result_ids, relevant_ids, len(relevant_ids)))
            MAP.append(self.average_precision(result_ids, relevant_ids))
        return [sum(MP3)/len(MP3), sum(MPR)/len(MPR), sum(MAP)/len(MAP)]
            

    def precision_at_k(self, result_ids, relevant_ids, k):
        """
        Compute the measure P@k for the given list of result ids as it was
        returned by the inverted index for a single query, and the given set of
        relevant document ids.

        Note that the relevant document ids are 1-based (as they reflect the
        line number in the dataset file).

        >>> evaluator = Evaluate()
        >>> evaluator.precision_at_k([5, 3, 6, 1, 2], {1, 2, 5, 6, 7, 8}, k=0)
        0
        >>> evaluator.precision_at_k([5, 3, 6, 1, 2], {1, 2, 5, 6, 7, 8}, k=4)
        0.75
        >>> evaluator.precision_at_k([5, 3, 6, 1, 2], {1, 2, 5, 6, 7, 8}, k=8)
        0.5
        """
        precision = 0
        for a in result_ids[:k]:
            if a in relevant_ids:
                precision +=1
        return precision/k

    def average_precision(self, result_ids, relevant_ids):
        """
        Compute the average precision (AP) for the given list of result ids as
        it was returned by the inverted index for a single query, and the given
        set of relevant document ids.

        Note that the relevant document ids are 1-based (as they reflect the
        line number in the dataset file).

        >>> evaluator = Evaluate()
        >>> evaluator.average_precision([7, 17, 9, 42, 5], {5, 7, 12, 42})
        0.525
        """
        R_list = []
        relevant_ids = set(relevant_ids)
        for a in relevant_ids:
            if a in result_ids:
                R_list.append(result_ids.index(a))
        result = []
        for a in R_list:
            result.append(self.precision_at_k(result_ids, relevant_ids, a+1))
        if len(result) != 0: 
            return sum(result)/len(relevant_ids) 
        else:
            return 0
            
        
                


def main():
    """
    Constructs an inverted index from the given dataset and evaluates the
    inverted index against the given benchmark.
    """
    # Parse the command line arguments.
    if len(sys.argv) != 3:
        print("Usage: python3 %s <file> <benchmark> [<b>] [<k>]" % sys.argv[0])
        sys.exit()

    data_file = sys.argv[1]
    benchmark_file = sys.argv[2]
    evaluate = Evaluate()
    ii = InvertedIndex()
    ii.build_from_file(data_file, 0.1, 0.95)
    
    benchmarks = evaluate.read_benchmark(benchmark_file)
    result = evaluate.evaluate(ii, benchmarks)
    a = np.array(result)
    b = np.array([1,1,1])
    
    distance = np.linalg.norm(a - b)
    
    
    print(result)


if __name__ == "__main__":
    main()