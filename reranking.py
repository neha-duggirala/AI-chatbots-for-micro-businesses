from langchain_typesafe import Score, TypeSafeClassifier
from dotenv import load_dotenv
from plot_score_answer import plot_score_answer
load_dotenv()
classifier = TypeSafeClassifier(name="Reranker")

state = "What time will my milk be delivered"
top_k_retreived = [
    """* **Transit Time:** It takes **24 to 36 hours** from the milking process at the farm to reach your doorstep, ensuring maximum freshness compared to conventional 72–96 hour supply chains.
* **Delivery Hours:** Daily morning doorstep delivery between **4:30 AM and 7:00 AM**.
* **Order Cut-Off:** Orders for next-day delivery must be placed before **9:00 PM**.""",
    """* **No Artificial Preservatives:** Because the milk is non-homogenized, non-standardized, and contains zero chemical stabilizers, it may appear thin or natural in consistency.
* **Cold Chain Dependency:** The milk must be collected immediately upon delivery and refrigerated promptly. Leaving it un-refrigerated for extended periods may cause natural curdling due to the absence of artificial preservatives.""",
]


response = classifier.invoke({
    "state": state,
    "questions":{
        "priority": Score(
            instructions = "Rank the following answers based on their relevance to the question.",
            criteria=top_k_retreived
        )
    }
})
print(response)


answer = response.answers["priority"]
print(answer)
plot_score_answer(answer, title="Ticket priority")