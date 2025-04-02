import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score
import tempfile
import pickle
import math
from collections import OrderedDict

def length_to_distance(length: float) -> float:
    return (2 - 2*math.cos(length))**(1/2)

def distance_to_length(distance: float) -> float:
    return math.acos((2-distance**2)/2)

def score_to_distance(score: float) -> float:
    return similarity_to_distance(score_to_similarity(score))

def distance_to_similarity(distance: float) -> float:
    return (2 - distance ** 2) / 2

# set the cohort radius
# for theta = pi/2 -> cohort_radius = sqrt(2), for theta = pi/4 -> cohort_radius = sqrt(2 - sqrt(2)), for theta = pi/3 -> cohort_radius = 1
cohort_radius = 1
query_radius = 1

# load the 2D ids and embeddings
# create profiles dictionary
profile_ids = list([f"P{i+1}" for i in range(16)])
profile_embeddings = list([[1, 0], [3**(1/2)/2, 1/2], [1/2**(1/2), 1/2**(1/2)], [1/2, 3**(1/2)/2], 
                      [0, 1], [-1/2, 3**(1/2)/2], [-1/2**(1/2), 1/2**(1/2)], [-3**(1/2)/2, 1/2],
                      [-1, 0], [-3**(1/2)/2, -1/2], [-1/2**(1/2), -1/2**(1/2)], [-1/2, -3**(1/2)/2], 
                      [0, -1], [1/2, -3**(1/2)/2], [1/2**(1/2), -1/2**(1/2)], [3**(1/2)/2, -1/2]
                     ])

profiles = {"profile_id": profile_ids, "profile_embedding": profile_embeddings}

# create cohorts for QII, QIII, QIV
cohort_ids = [f"C{i+1}" for i in range(3)]
cohort_embeddings = [[1/2**(1/2), 1/2**(1/2)], [-1/2**(1/2), 1/2**(1/2)], [-1/2**(1/2), -1/2**(1/2)]]
cohorts = {"cohort_id": cohort_ids, "cohort_embedding": cohort_embeddings}

# create queries: for this example query embeddings = [cos(theta), sin(theta)] for theta = pi/6, 5pi/6, pi/3
query_ids = [f"Q{i+1}" for i in range(3)]
query_embeddings = [[1, 0], [-3**(1/2)/2, 1/2], [1/2, -3**(1/2)/2]]
queries = {"query_id": query_ids, "query_embedding": query_embeddings}


# cohort assignment algorithm
# tag profiles with cohorts in range, tag cohorts with profiles in range
profile_cohort_ids = []
cohort_profile_ids = {cohort_id: [] for cohort_id in cohort_ids}  # Dictionary to track profiles for each cohort

def multiple_cohort_assignment(cohorts, profiles, cohort_radius):
    cohort_ids = np.array(cohorts["cohort_id"]) 
    cohort_embeddings = np.array(cohorts["cohort_embedding"])
    profile_ids = np.array(profiles["profile_id"]) 
    profile_embeddings = np.array(profiles["profile_embedding"])
    print(f"Assigning {len(profile_ids)} profiles to {len(cohort_ids)} cohorts...")

    knn = NearestNeighbors(algorithm='brute', metric="euclidean", n_jobs=-1)
    knn.fit(cohort_embeddings)
    cohort_distances, cohort_indices = knn.kneighbors(profile_embeddings, n_neighbors=3, return_distance=True)
    
    # Properly initialize cohort_profile_ids dictionary
    cohort_profile_ids = {cohort_id: [] for cohort_id in cohort_ids}
    profile_cohort_ids = []
    
    for profile_idx, (distances, indices) in enumerate(zip(cohort_distances, cohort_indices)):
        assigned_cohort_ids = []
        for distance, cohort_idx in zip(distances, indices):
            if distance <= cohort_radius:
                cohort_id = cohort_ids[cohort_idx]
                assigned_cohort_ids.append(cohort_id)
                cohort_profile_ids[cohort_id].append(profile_ids[profile_idx])  # Track profiles in respective cohorts
        profile_cohort_ids.append(assigned_cohort_ids)

    tagged_profiles = [
        {"profile_id": pid, "profile_embedding": pembed, "cohort_ids": cid}
        for pid, pembed, cid in zip(profile_ids, profile_embeddings, profile_cohort_ids)
    ]

    tagged_cohorts = [
        {"cohort_id": cid, "cohort_embedding": cembed, "profile_ids": cohort_profile_ids[cid]}
        for cid, cembed in zip(cohort_ids, cohort_embeddings)
    ]

    return tagged_profiles, tagged_cohorts

tagged_profiles, tagged_cohorts = multiple_cohort_assignment(cohorts, profiles, cohort_radius)

# cohort querying algorithms
# brute force for precision and recall measures
def brute_force_search(queries, tagged_profiles, min_cosine_similarity):
    query_embeddings = np.array(queries["query_embedding"])
    profile_embeddings = np.stack([profile["profile_embedding"] for profile in tagged_profiles])    
    cosine_similarities = cosine_similarity(query_embeddings, profile_embeddings)    
    return [len(profile_embeddings)] * len(query_embeddings), (cosine_similarities >= min_cosine_similarity).astype(int)


def cohort_search(queries, tagged_cohorts, query_radius, cohort_radius):
    cohort_ids = [cohort["cohort_id"] for cohort in tagged_cohorts]
    cohort_embeddings = np.stack([cohort["cohort_embedding"] for cohort in tagged_cohorts]) if tagged_cohorts else np.array([])
    query_ids = queries["query_id"]
    query_embeddings = np.array(queries["query_embedding"])
    
    print(f"Querying cohorts...")
    if cohort_embeddings.size > 0:
        cohort_knn = NearestNeighbors(algorithm='brute', metric="euclidean", n_jobs=-1)
        cohort_knn.fit(cohort_embeddings)
        all_cohort_indices = cohort_knn.radius_neighbors(
            query_embeddings,
            radius=length_to_distance(distance_to_length(query_radius) + distance_to_length(cohort_radius)),
            return_distance=False
        )
    else:
        all_cohort_indices = [[] for _ in range(len(query_embeddings))]
    
    print(f"There are {[len(cohort_indices) for cohort_indices in all_cohort_indices]} cohorts matched to each query, respectively.")
    
    return cohort_ids, all_cohort_indices


def collect_profiles_to_query(tagged_profiles, cohort_ids, all_cohort_indices, query_count):
    profiles_to_query = {query_idx: {"profile_id": [], "profile_embedding": []} for query_idx in range(query_count)}

    for query_idx in range(query_count):
        cohort_matched_profiles = OrderedDict()  # Preserves insertion order
        
        for cohort_idx in all_cohort_indices[query_idx]:
            cohort_id = cohort_ids[cohort_idx]
            inner_profiles = [(profile["profile_id"], profile["profile_embedding"]) 
                              for profile in tagged_profiles if cohort_id in profile["cohort_ids"]]
            
            for profile_id, embedding in inner_profiles:
                if profile_id not in cohort_matched_profiles:
                    cohort_matched_profiles[profile_id] = embedding
        
        # Extend the lists while maintaining order
        profiles_to_query[query_idx]["profile_id"].extend(cohort_matched_profiles.keys())
        profiles_to_query[query_idx]["profile_embedding"].extend(cohort_matched_profiles.values())

    # Retrieve uncohorted profiles
    uncohorted_profiles = [(profile["profile_id"], profile["profile_embedding"]) 
                           for profile in tagged_profiles if not profile["cohort_ids"]]

    for query_idx in range(query_count):
        for profile_id, embedding in uncohorted_profiles:
            profiles_to_query[query_idx]["profile_id"].append(profile_id)
            profiles_to_query[query_idx]["profile_embedding"].append(embedding)

    inner_profile_calculation_counts = [len(profiles_to_query[q_idx]['profile_id']) for q_idx in range(query_count)]

    print(f"Profiles collected per query: {inner_profile_calculation_counts}")
    print(f"profiles_to_query: ")
    print(f"{profiles_to_query}")

    return profiles_to_query


def profile_search(queries, tagged_profiles, cohort_ids, all_cohort_indices, query_radius):
    query_ids = queries["query_id"]
    query_embeddings = np.array(queries["query_embedding"])
    
    print(f"Querying profiles...")

    # Collect profiles to query
    profiles_to_query = collect_profiles_to_query(tagged_profiles, cohort_ids, all_cohort_indices, len(query_ids))
    
    # Count of profiles in each cohort per query
    cohort_calculation_counts = [len(cohort_ids) + len(profiles_to_query[q_idx]["profile_id"]) for q_idx in range(len(query_ids))]
    print(f"Cohort calculation counts per query: {cohort_calculation_counts}")
    
    # Initialize the list to hold profile ids in range for each query
    in_range_profile_ids = [[] for _ in range(len(query_ids))]
    
    # Query processing: find profiles in range for each query
    for query_idx in range(len(query_ids)):
        profiles = profiles_to_query[query_idx]
        print(f"profiles_to_query for query_idx {query_idx}: ")
        print(f"{profiles}")
        embeddings = np.array(profiles["profile_embedding"]) if profiles["profile_embedding"] else np.array([])

        if embeddings.size > 0:
            # Perform the nearest neighbor search
            profile_knn = NearestNeighbors(algorithm="brute", metric="euclidean", n_jobs=-1)
            profile_knn.fit(embeddings)
            all_inner_profile_indices = profile_knn.radius_neighbors([query_embeddings[query_idx]], radius=query_radius, return_distance=False)[0]
            print()
            print(f"all_inner_profile_indices for query_idx {query_idx}: ")
            print(all_inner_profile_indices)
            print()
            in_range_profile_ids[query_idx] = [profiles["profile_id"][i] for i in all_inner_profile_indices]
    
    print(f"in_range_profile_ids: ")
    print(in_range_profile_ids)
    print()
    # Map profile_id to index for fast lookup
    profile_id_to_index = {profile["profile_id"]: idx for idx, profile in enumerate(tagged_profiles)}
    
    # Initialize the result array (query x profiles)
    cohort_query_results = np.zeros((len(query_ids), len(tagged_profiles)), dtype=int)

    # For each query, create a set of in-range profile IDs for fast lookup
    for query_idx in range(len(query_ids)):
        matched_profile_ids = in_range_profile_ids[query_idx]
        for profile_idx, profile in enumerate(tagged_profiles):
            # Check if the profile_id is in the matched_profile_ids
            if profile["profile_id"] in matched_profile_ids:
                cohort_query_results[query_idx, profile_idx] = 1
            else:
                cohort_query_results[query_idx, profile_idx] = 0
    
    return cohort_calculation_counts, cohort_query_results

brute_force_calculation_counts, brute_force_query_results = brute_force_search(queries, tagged_profiles, distance_to_similarity(query_radius))

cohort_ids, all_cohort_indices = cohort_search(queries, tagged_cohorts, query_radius, cohort_radius)

cohort_calculation_counts, cohort_query_results = profile_search(queries, tagged_profiles, cohort_ids, all_cohort_indices, query_radius)


def score_results(y_trues: list[list[int]], y_preds: list[list[int]]):
    y_trues_counts = [len(y_trues[query_idx]) for query_idx in range(len(y_trues))]
    y_preds_counts = [len(y_preds[query_idx]) for query_idx in range(len(y_preds))]
    print(f"Counts of y_trues by query: {y_trues_counts}.")
    print(f"Counts of y_preds by query: {y_preds_counts}.")
    precisions = [precision_score(y_trues[query_idx], y_preds[query_idx]) for query_idx in range(len(y_trues))]
    recalls = [recall_score(y_trues[query_idx], y_preds[query_idx]) for query_idx in range(len(y_trues))]
    return precisions, recalls

precisions, recalls = score_results(brute_force_query_results, cohort_query_results)


print(f"brute_force_calculation_counts: {brute_force_calculation_counts}")
print(f"brute_force_query_results: ")
print(f"{brute_force_query_results}")
print()
print(f"cohort_calculation_counts: {cohort_calculation_counts}")
print(f"cohort_query_results: ")
print(f"{cohort_query_results}")
print()
print(f"precisions: {precisions}")
print(f"recalls: {recalls}")