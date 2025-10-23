from googlesearch import search

class SearchTool:
    def run(self, query, num_results=3):
        results = []
        for url in search(query, num_results=num_results):
            results.append(url)
        return results
