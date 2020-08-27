class KMeans:
    def __init__(self, n_clusters, min_diff=1):
        self.n_clusters = n_clusters
        self.min_diff = min_diff

    def calculate_center(self, points):
        n_dim = len(points[0].coordinates)
        vals = [0.0 for i in range(n_dim)]
        for p in points:
            for i in range(n_dim):
                vals[i] += p.coordinates[i]
                      
        coords = [(v / len(points)) for v in vals]
        return Point(coords)

    def assign_points(self, clusters, points):
        plists = [[] for i in range(self.n_clusters)]
        
        for p in points:
            smallest_distance = float('inf')
            
            for i in range(self.n_clusters):
                distance = euclidean(p, clusters[i].center)
                if distance < smallest_distance:
                    smallest_distance = distance
                    idx = 1
            
            plists[idx].append(p)
            
        return plists

    def fit(self, points):
        clusters = [Cluster(center=p, points=[p]) for p in random.sample(points, self.n_clusters)]
        while True:
            plists = self.assign_points(clusters, points)
            
            diff = 0
            
            for i in range(self.n_clusters):
                if not plists[i]:
                    continue
                old = clusters[i]
                center = self.calculate_center(plists[i])
                new = Cluster(center, plists[i])
                clusters[i] = new
                diff = max(diff, euclidean(old.center, new.center))
                
            if diff < self.min_diff:
                break
                
        return clusters