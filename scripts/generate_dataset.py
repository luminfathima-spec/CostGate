"""
Dataset generator for CostGate Phase 2 evaluation benchmark.
Generates 60 questions (15 per category), stratified 70/30 train/test split with seed 42.
Also generates eval/review_sample.csv with 10 random rows.
"""

import csv
import random
from pathlib import Path

# 60 questions across 4 categories (15 per category)
DATA = [
    # 1. simple_fact (15)
    {"id": 1, "category": "simple_fact", "question": "What is the capital of France?", "key_points": "Paris | capital of France | located on the Seine River"},
    {"id": 2, "category": "simple_fact", "question": "What is the chemical symbol for gold?", "key_points": "Au | Latin word Aurum | atomic number 79"},
    {"id": 3, "category": "simple_fact", "question": "Who wrote the play Romeo and Juliet?", "key_points": "William Shakespeare | English playwright | tragedy"},
    {"id": 4, "category": "simple_fact", "question": "What is the largest planet in our solar system?", "key_points": "Jupiter | gas giant | largest planet"},
    {"id": 5, "category": "simple_fact", "question": "What is the freezing point of water in Celsius at standard atmospheric pressure?", "key_points": "0 degrees Celsius | 0 °C | standard atmospheric pressure"},
    {"id": 6, "category": "simple_fact", "question": "In which year did the Apollo 11 mission land the first humans on the Moon?", "key_points": "1969 | July 20, 1969 | Neil Armstrong and Buzz Aldrin"},
    {"id": 7, "category": "simple_fact", "question": "What is the hardest naturally occurring mineral on Earth?", "key_points": "Diamond | carbon allotrope | Mohs hardness scale of 10"},
    {"id": 8, "category": "simple_fact", "question": "Which ocean is the largest by surface area?", "key_points": "Pacific Ocean | covers more than 30% of Earth | largest ocean"},
    {"id": 9, "category": "simple_fact", "question": "What organelle is commonly referred to as the powerhouse of the cell?", "key_points": "Mitochondria | cellular respiration | ATP production"},
    {"id": 10, "category": "simple_fact", "question": "How many chromosomes do typical human somatic cells possess?", "key_points": "46 chromosomes | 23 pairs | diploid count"},
    {"id": 11, "category": "simple_fact", "question": "What is the exact speed of light in a vacuum in meters per second?", "key_points": "299,792,458 m/s | ~3x10^8 m/s | fundamental constant c"},
    {"id": 12, "category": "simple_fact", "question": "What is the most abundant gas in Earth's atmosphere by volume?", "key_points": "Nitrogen | accounts for approximately 78% | N2"},
    {"id": 13, "category": "simple_fact", "question": "Who painted the Mona Lisa?", "key_points": "Leonardo da Vinci | Italian Renaissance polymath | painted in early 16th century"},
    {"id": 14, "category": "simple_fact", "question": "What is the official currency of Japan?", "key_points": "Japanese Yen | symbol ¥ | ISO code JPY"},
    {"id": 15, "category": "simple_fact", "question": "Which desert is the largest hot desert in the world?", "key_points": "Sahara Desert | located in Northern Africa | covers over 9 million square kilometers"},

    # 2. explanation (15)
    {"id": 16, "category": "explanation", "question": "How do vaccines stimulate the human immune system to protect against disease?", "key_points": "introduce antigen or weakened pathogen | stimulate antibody production | create memory B and T cells | provide rapid response upon reinfection"},
    {"id": 17, "category": "explanation", "question": "Why does the clear daytime sky appear blue to observers on Earth?", "key_points": "Rayleigh scattering | sunlight enters atmosphere | shorter blue wavelengths scatter more than red | sunlight interacts with gas molecules"},
    {"id": 18, "category": "explanation", "question": "Explain the physical mechanism of the greenhouse effect on Earth.", "key_points": "solar radiation warms Earth surface | surface emits infrared radiation | greenhouse gases absorb infrared | re-radiate heat back toward surface"},
    {"id": 19, "category": "explanation", "question": "How does an airplane wing generate aerodynamic lift during forward flight?", "key_points": "airfoil shape | pressure differential between upper and lower surfaces | faster airflow over curved top creates lower pressure | downward deflection of airflow"},
    {"id": 20, "category": "explanation", "question": "How does the Domain Name System (DNS) resolve a hostname into an IP address?", "key_points": "recursive resolver receives query | queries root nameserver | queries TLD nameserver (.com) | queries authoritative nameserver to obtain IP"},
    {"id": 21, "category": "explanation", "question": "What physical forces cause ocean tides on Earth?", "key_points": "gravitational pull of the Moon | gravitational pull of the Sun | tidal bulges formed on near and far sides | Earth rotates through bulges"},
    {"id": 22, "category": "explanation", "question": "Explain how photosynthesis converts light energy into chemical energy in plants.", "key_points": "occurs in chloroplasts with chlorophyll | absorbs sunlight and water to produce oxygen (light reactions) | fixes carbon dioxide into glucose (Calvin cycle) | generates ATP and NADPH"},
    {"id": 23, "category": "explanation", "question": "Why do seasonal changes occur on Earth throughout the year?", "key_points": "axial tilt of Earth (approx 23.5 degrees) | orbit around the Sun | varying angle of solar incidence | changes in daylight hours and solar energy intensity"},
    {"id": 24, "category": "explanation", "question": "How does a microwave oven heat food?", "key_points": "generates electromagnetic waves at 2.45 GHz | dielectric heating of water molecules | polar water molecules rotate rapidly to align with field | molecular friction produces thermal heat"},
    {"id": 25, "category": "explanation", "question": "Explain how soap molecules remove oil and grease from skin or clothing.", "key_points": "amphipathic structure | hydrophobic non-polar tail attaches to grease | hydrophilic polar head interacts with water | forms micelles that suspend grease in water"},
    {"id": 26, "category": "explanation", "question": "What is the physiological role of insulin in the human body?", "key_points": "hormone produced by beta cells in pancreas | facilitates glucose uptake by cells | stimulates glycogen synthesis in liver | lowers blood glucose levels"},
    {"id": 27, "category": "explanation", "question": "How does active noise-canceling headphone technology reduce ambient sound?", "key_points": "microphone detects external noise | internal processor creates inverted sound wave (anti-phase) | destructive interference cancels noise waves | speaker outputs anti-noise signal"},
    {"id": 28, "category": "explanation", "question": "Explain the biological function of DNA replication and how it maintains sequence fidelity.", "key_points": "semi-conservative replication | DNA polymerase synthesizes complementary strands | proofreading exonuclease activity catches mismatches | creates identical DNA copies for cell division"},
    {"id": 29, "category": "explanation", "question": "How does a thermos (vacuum flask) prevent heat loss from hot liquids?", "key_points": "vacuum layer eliminates conduction and convection | reflective inner silver coating reduces radiation | tight stopper prevents evaporative heat loss | maintains temperature over time"},
    {"id": 30, "category": "explanation", "question": "Why do oil and water not naturally mix together?", "key_points": "water is polar with strong hydrogen bonds | oil is non-polar and hydrophobic | energetically unfavorable to disrupt water-water bonds | phase separation minimizes free energy"},

    # 3. reasoning (15)
    {"id": 31, "category": "reasoning", "question": "A train leaves City A at 60 mph toward City B, 180 miles away. Another train leaves City B at the same time toward City A at 30 mph. How many hours until they meet?", "key_points": "combined speed is 90 mph | total distance is 180 miles | time = distance / combined speed | 2 hours"},
    {"id": 32, "category": "reasoning", "question": "If 5 machines take 5 minutes to make 5 widgets, how many minutes will 100 machines take to make 100 widgets?", "key_points": "1 machine takes 5 minutes to make 1 widget | machines operate in parallel | 100 machines make 100 widgets simultaneously | 5 minutes"},
    {"id": 33, "category": "reasoning", "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents?", "key_points": "bat + ball = 1.10 | bat = ball + 1.00 | 2 * ball + 1.00 = 1.10 | ball = 5 cents"},
    {"id": 34, "category": "reasoning", "question": "In a room of 30 people, if each person shakes hands with every other person exactly once, how many total handshakes occur?", "key_points": "combination formula n*(n-1)/2 | 30 * 29 / 2 | 435 total handshakes"},
    {"id": 35, "category": "reasoning", "question": "You have two ropes. Each takes exactly 60 minutes to burn from end to end, but they burn unevenly. How do you measure 45 minutes using only these two ropes and a lighter?", "key_points": "light rope 1 from both ends and rope 2 from one end | when rope 1 burns out, exactly 30 minutes elapsed | at that exact moment, light the other end of rope 2 | remaining half burns in 15 minutes, totaling 45 minutes"},
    {"id": 36, "category": "reasoning", "question": "If all Bloops are Razzies, and all Razzies are Lizzies, are all Bloops necessarily Lizzies? Explain why or why not.", "key_points": "yes, all Bloops are Lizzies | transitivity of subset relations | Bloops are a subset of Razzies | Razzies are a subset of Lizzies"},
    {"id": 37, "category": "reasoning", "question": "A store offers a 20% discount on an item, followed by an additional 10% discount on the discounted price. What is the total effective percentage discount from the original price?", "key_points": "first discount leaves 80% of original | 10% off 80% is 8% discount | remaining price is 72% | total effective discount is 28%"},
    {"id": 38, "category": "reasoning", "question": "A water tank has two inlet pipes. Pipe A fills the tank in 4 hours, and Pipe B fills it in 6 hours. Working together, how many hours will they take to fill the tank?", "key_points": "rate of A is 1/4 per hour | rate of B is 1/6 per hour | combined rate is 5/12 per hour | time to fill is 12/5 hours or 2.4 hours (2 hours 24 minutes)"},
    {"id": 39, "category": "reasoning", "question": "A father is currently 4 times as old as his son. In 20 years, the father will be twice as old as his son. How old is the son now?", "key_points": "F = 4S | F + 20 = 2*(S + 20) | 4S + 20 = 2S + 40 | son is 10 years old (father is 40)"},
    {"id": 40, "category": "reasoning", "question": "Three fair coins are flipped simultaneously. What is the probability of getting at least two heads?", "key_points": "total outcomes is 2^3 = 8 | outcomes with >=2 heads: HHH, HHT, HTH, THH | 4 favorable outcomes | probability is 4/8 or 1/2 (50%)"},
    {"id": 41, "category": "reasoning", "question": "If it takes 8 workers 6 days to build a brick wall, how many days will 12 workers take to build the same wall, assuming equal productivity?", "key_points": "total work is 8 * 6 = 48 worker-days | days = 48 / 12 workers | 4 days"},
    {"id": 42, "category": "reasoning", "question": "A car travels the first 60 miles of a trip at 30 mph and the next 60 miles at 60 mph. What is the average speed for the entire 120-mile trip?", "key_points": "time for first leg is 2 hours | time for second leg is 1 hour | total time is 3 hours | average speed = 120 / 3 = 40 mph"},
    {"id": 43, "category": "reasoning", "question": "A box contains 5 red balls and 3 green balls. If 2 balls are drawn at random without replacement, what is the probability that both balls are red?", "key_points": "probability of first red is 5/8 | probability of second red is 4/7 | (5/8) * (4/7) = 20/56 | simplified to 5/14"},
    {"id": 44, "category": "reasoning", "question": "There are 3 boxes: one labeled 'Apples', one 'Oranges', and one 'Apples and Oranges'. All three labels are incorrect. You can pick one fruit from one box. Which box do you pick from to determine the contents of all three boxes?", "key_points": "pick from the box labeled 'Apples and Oranges' | since label is wrong, it contains only the fruit you draw (e.g., Apple) | the box labeled with the other fruit must be the mix | the remaining box is the other single fruit"},
    {"id": 45, "category": "reasoning", "question": "Sally has 3 brothers and each brother has 2 sisters. How many sisters does Sally have?", "key_points": "each brother has 2 sisters | the sisters are Sally and 1 other sister | there are 2 sisters in the family | Sally has 1 sister"},

    # 4. technical (15)
    {"id": 46, "category": "technical", "question": "Explain the role and impact of the Global Interpreter Lock (GIL) in CPython.", "key_points": "mutex protecting access to Python objects | prevents multiple native threads from executing Python bytecode simultaneously | simplifies memory management | limits CPU-bound multi-threading performance"},
    {"id": 47, "category": "technical", "question": "What are the four ACID properties in database management systems?", "key_points": "Atomicity (all or nothing) | Consistency (valid state transitions) | Isolation (concurrent transactions execute independently) | Durability (committed data survives system crashes)"},
    {"id": 48, "category": "technical", "question": "What is the worst-case and average-case time complexity of Quicksort, and what causes the worst-case?", "key_points": "average-case O(n log n) | worst-case O(n^2) | worst-case occurs with unbalanced partitions (e.g. sorted array with poor pivot choice) | partition-based divide-and-conquer"},
    {"id": 49, "category": "technical", "question": "Describe the three-way handshake process used by TCP to establish a reliable connection.", "key_points": "Client sends SYN packet | Server replies with SYN-ACK packet | Client sends ACK packet | connection is established with synchronized sequence numbers"},
    {"id": 50, "category": "technical", "question": "What is the difference between a process and a thread in modern operating systems?", "key_points": "process has independent virtual address space and memory | threads within a process share the same memory space | processes require IPC to communicate | thread context switching has lower overhead than process context switching"},
    {"id": 51, "category": "technical", "question": "Explain the four Coffman conditions necessary for a deadlock to occur in an operating system.", "key_points": "Mutual exclusion | Hold and wait | No preemption | Circular wait"},
    {"id": 52, "category": "technical", "question": "What is the time complexity of searching in a balanced Binary Search Tree (AVL or Red-Black Tree) versus an unbalanced tree?", "key_points": "balanced BST worst-case is O(log n) | unbalanced tree degrades to O(n) equivalent to a linked list | height of balanced tree is bounded by O(log n)"},
    {"id": 53, "category": "technical", "question": "Explain the core architectural differences between REST and GraphQL APIs.", "key_points": "REST uses fixed endpoints returning fixed data structures | GraphQL uses a single endpoint allowing clients to query specific fields | REST often suffers from over-fetching or under-fetching | GraphQL uses strongly-typed schemas and resolvers"},
    {"id": 54, "category": "technical", "question": "What is the difference between synchronous and asynchronous I/O in software engineering?", "key_points": "synchronous I/O blocks calling thread until operation completes | asynchronous I/O allows calling thread to continue executing | async uses event loops or callbacks/promises | async enables high concurrency without thread-per-connection overhead"},
    {"id": 55, "category": "technical", "question": "Explain how public-key (asymmetric) cryptography enables both confidentiality and digital signatures.", "key_points": "pair of mathematically linked keys: public and private | encryption with public key ensures confidentiality (only private key decrypts) | encryption with private key creates digital signature (verified with public key) | solves key distribution problem"},
    {"id": 56, "category": "technical", "question": "What is an index in a relational database, and what is the trade-off of adding too many indexes?", "key_points": "data structure (often B-tree) that accelerates data retrieval | reduces query lookup time from O(n) to O(log n) | increases storage overhead | degrades write performance (INSERT, UPDATE, DELETE must update indexes)"},
    {"id": 57, "category": "technical", "question": "What is a memory leak, and what is the primary difference between how manual memory management and garbage collection handle memory?", "key_points": "memory allocated but no longer reachable or freed | manual requires explicit free/delete (e.g., C/C++) | garbage collection automatically detects and reclaims unreachable objects | GC introduces runtime overhead or pauses"},
    {"id": 58, "category": "technical", "question": "Explain the purpose of Docker containers and how they differ from traditional Virtual Machines (VMs).", "key_points": "containers share the host operating system kernel | VMs run a full guest operating system on top of a hypervisor | containers are lightweight with faster startup times | VMs provide stronger hardware-level isolation"},
    {"id": 59, "category": "technical", "question": "What is Cross-Origin Resource Sharing (CORS) and why does the browser enforce it?", "key_points": "browser security mechanism using HTTP headers | restricts web pages from making requests to a different domain | protects against CSRF and unauthorized cross-domain data access | servers declare allowed origins via Access-Control-Allow-Origin"},
    {"id": 60, "category": "technical", "question": "Explain the difference between optimistic concurrency control and pessimistic concurrency control in database transactions.", "key_points": "pessimistic locks data records upon access to prevent conflicts | optimistic assumes conflicts are rare and verifies data unchanged at commit time | pessimistic reduces concurrency in read-heavy systems | optimistic rolls back and retries if version conflict detected"},
]

def generate_datasets():
    # Stratified 70/30 split with fixed seed 42
    # In each category of 15 items:
    # 2 categories get 11 train / 4 test, 2 categories get 10 train / 5 test -> 42 train (70%), 18 test (30%)
    rng = random.Random(42)

    categories = ["simple_fact", "explanation", "reasoning", "technical"]
    category_items = {c: [] for c in categories}
    for item in DATA:
        category_items[item["category"]].append(item)

    # Deterministically shuffle each category and assign split
    # Category splits:
    # simple_fact: 11 train, 4 test
    # explanation: 10 train, 5 test
    # reasoning: 11 train, 4 test
    # technical: 10 train, 5 test
    # Total = 42 train, 18 test (exactly 70.0% / 30.0%)
    split_counts = {
        "simple_fact": 11,
        "explanation": 10,
        "reasoning": 11,
        "technical": 10,
    }

    final_dataset = []
    for cat in categories:
        items = category_items[cat]
        # Sort by id first for consistency, then shuffle with seed
        items.sort(key=lambda x: x["id"])
        indices = list(range(len(items)))
        rng.shuffle(indices)
        train_indices = set(indices[:split_counts[cat]])

        for idx, item in enumerate(items):
            item_copy = dict(item)
            item_copy["split"] = "train" if idx in train_indices else "test"
            final_dataset.append(item_copy)

    # Sort final dataset by ID
    final_dataset.sort(key=lambda x: x["id"])

    # Verify counts
    total = len(final_dataset)
    train_count = sum(1 for x in final_dataset if x["split"] == "train")
    test_count = sum(1 for x in final_dataset if x["split"] == "test")
    assert total == 60, f"Expected 60, got {total}"
    assert train_count == 42, f"Expected 42 train (70%), got {train_count}"
    assert test_count == 18, f"Expected 18 test (30%), got {test_count}"

    for cat in categories:
        cat_items = [x for x in final_dataset if x["category"] == cat]
        assert len(cat_items) == 15, f"Expected 15 for {cat}, got {len(cat_items)}"

    # Write eval/dataset.csv
    dataset_path = Path(__file__).resolve().parent.parent / "eval" / "dataset.csv"
    with open(dataset_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "category", "key_points", "split"])
        writer.writeheader()
        writer.writerows(final_dataset)
    print(f"Saved {len(final_dataset)} rows to {dataset_path}")

    # Write eval/review_sample.csv with 10 random rows (using seed 42)
    sample_rng = random.Random(42)
    sample_rows = sample_rng.sample(final_dataset, 10)
    sample_rows.sort(key=lambda x: x["id"])

    sample_path = Path(__file__).resolve().parent.parent / "eval" / "review_sample.csv"
    with open(sample_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "category", "key_points", "split"])
        writer.writeheader()
        writer.writerows(sample_rows)
    print(f"Saved 10 sample rows to {sample_path}")

if __name__ == "__main__":
    generate_datasets()
