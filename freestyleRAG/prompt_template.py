PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["🌑", "🌒", "🌓", "🌔", "🌕"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]

PROMPTS["triples_extraction"] = """
-Goal-

Your task is to identify all the entities in a given text, extracting the relationships between the entities if they are related to each other based on your understanding of the knowledge graph. Detailed steps are given below.

-Steps-

1. Identify all the entities and the relationships that exist between them. For the relationship pairs between the entities, extract the following information:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: use a short statement to explain the relationship between the source entity and target entity.
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

2. If the entity found in the text is not associated with any other entity, then the name of the source entity is the same as the target entity, and the relationship description becomes a simple description of this special entity, find them and extract the following information:
- source_entity: name of this special entity
- target_entity: name of this special entity
- relationship_description: use a short statement to describe this entity.
- relationship_strength: a numeric score representing the importance of this special entity in the text
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return the output in English as a single list containing all the relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

Here are some examples:

Example 1:

Text: TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.


Output:
("relationship"{tuple_delimiter}"c"{tuple_delimiter}"Global Exchange"{tuple_delimiter}"listed its stock on"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"TechGlobal (TG)"{tuple_delimiter}"Newly Listed Companies"{tuple_delimiter}"debut is not indicative of"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"TechGlobal (TG)"{tuple_delimiter}"Vision Holdings"{tuple_delimiter}"was taken private by in 2014"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"TechGlobal (TG)"{tuple_delimiter}"Premium Smartphones"{tuple_delimiter}"powers 85% of"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"IPO Experts"{tuple_delimiter}"IPO Experts"{tuple_delimiter}"provide insight on stock market behavior"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"TechGlobal IPO, stock performance, semiconductor industry, Vision Holdings, premium smartphones, public markets, IPO analysis"{tuple_delimiter}){completion_delimiter}


Example 2:

Text: while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths

Output:
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Frustration"{tuple_delimiter}"experiences"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"interacts with"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Authoritarian Certainty"{tuple_delimiter}"represents"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"opposes vision of control and order"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Cruz"{tuple_delimiter}"Control and Order"{tuple_delimiter}"represents"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"shows reverence towards"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"The Device"{tuple_delimiter}"Technology"{tuple_delimiter}"is associated with"{tuple_delimiter}4){record_delimiter}
("relationship"{tuple_delimiter}"Technology"{tuple_delimiter}"Game"{tuple_delimiter}"could change"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"All people"{tuple_delimiter}"All people"{tuple_delimiter}"were brought here by different paths"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Alex"{tuple_delimiter}"main character and observer"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Taylor"{tuple_delimiter}"authoritative figure with layers of complexity"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Jordan"{tuple_delimiter}"committed to discovery and unity"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Cruz"{tuple_delimiter}"Cruz"{tuple_delimiter}"symbolizes control and narrow vision"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"The Device"{tuple_delimiter}"The Device"{tuple_delimiter}"symbol of opportunity and transformation"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"power dynamics, discovery, control, transformation, competition, unspoken alliances"{tuple_delimiter}){completion_delimiter}

Example 3:

Text: I acknowledge with deep gratitude the dozens of friends, colleagues, and perfect strangers who helped me explore the benefits of indigenous diets. Without their contributions, this project could never have happened.\n\nFirst and foremost, I want to thank my beloved co-adventurer (and husband) Ross Levy. His patience and support allowed this book to move beyond a mere daydream.

I would also like to thank my children, Arlen and Emet Levy, who bravely tasted all the recipes and who never failed to give their honest opinions. I heartily thank Allison Fragakis for her excellent nutrition advice throughout this project and for spearheading the recipe-testing portion of this book.\n\nI am deeply grateful to my parents, Susan and David Miller, whose own wanderlust and love of eating and cooking first launched me into the world of travel adventure and indigenous foods and to my brother Sam Miller, the armchair nutritionist, who seems to know more than many professionals.

Output:
("relationship"{tuple_delimiter}"The Author"{tuple_delimiter}"Friends, Colleagues, and Perfect Strangers"{tuple_delimiter}"received help exploring indigenous diets from"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"The Author"{tuple_delimiter}"Ross Levy"{tuple_delimiter}"received patience and support from"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"The Author"{tuple_delimiter}"Arlen and Emet Levy"{tuple_delimiter}"received bravery and honest opinions about the recipes from"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"The Author"{tuple_delimiter}"Allison Fragakis"{tuple_delimiter}"received excellent nutrition advice and recipe-testing support from"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"The Author"{tuple_delimiter}"Susan and David Miller"{tuple_delimiter}"was inspired by their wanderlust and love of cooking"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"The Author"{tuple_delimiter}"Sam Miller"{tuple_delimiter}"benefited from his knowledge as an armchair nutritionist"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Friends, Colleagues, and Perfect Strangers"{tuple_delimiter}"friends, colleagues, and perfect strangers"{tuple_delimiter}"contributed to the exploration of indigenous diets"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Ross Levy"{tuple_delimiter}"Ross Levy"{tuple_delimiter}"co-adventurer and husband of the author"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Arlen and Emet Levy"{tuple_delimiter}"Arlen and Emet Levy"{tuple_delimiter}"children of the author who tasted recipes and gave opinions"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Allison Fragakis"{tuple_delimiter}"Allison Fragakis"{tuple_delimiter}"nutrition advisor and recipe tester for the project"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Susan and David Miller"{tuple_delimiter}"Susan and David Miller"{tuple_delimiter}"parents of the author with wanderlust and love for cooking"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Sam Miller"{tuple_delimiter}"Sam Miller"{tuple_delimiter}"brother of the author and an armchair nutritionist"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"indigenous diets, gratitude, family support, nutrition advice, recipe testing, wanderlust, cooking inspiration"{tuple_delimiter}){completion_delimiter}


-Real Data-

Text: {input_text}

Output:
"""

PROMPTS[
    "triples_continue_extraction"
] = """May be some relationships were missed in the last extraction. Add them below using the same format mentioned earlier:
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and similar in form to the given description.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################
Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:

"""

PROMPTS ["regenerate_description"] = """
-Global-

Your task is to complete the description in each given relationship or entity based on the given text, so as to form a complete and detailed sentence. Remember not to output entities or relationships that lack description. Detailed steps are given below.

1. For each entity, format it as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_description>)
- entity_name: Name of the entity
- entity_description: Comprehensive description of the entity's attributes and activities

2. For each relationship, format it as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>)
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

Here are some examples:

Example 1:

Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths

Entities:
None

Relationships:
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}){record_delimiter}

Output:
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."){completion_delimiter}

Example 2:

Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly

Entities:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}){record_delimiter}

Relationships:
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}){record_delimiter}

Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."){completion_delimiter}

Example 3:

Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation

Entities:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}){record_delimiter}

Relationships:
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}){record_delimiter}

Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."){completion_delimiter}

-Real Data-

Text:
{input_text}

Entities:
{entity_names}

Relationships:
{relationship_names}

Output:
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["qa_response"]="""You are an intelligent assistant capable of providing responses based on the relevant information provided. Only give me the answer and do not offer any explanations, defences or comments.\n\n### Data tables:### \n{context}\n\nOnly give me the answer entity based on the above information and output what you think is the most correct answer. If the given question requires multiple answers to be output, please separate them with commas. Never explain, justify, or add commentary.\n\nQuestion: {input}\nAnswer:"""

PROMPTS["qa_response_odqa"]="""You are an intelligent assistant capable of providing responses based on the relevant information provided. Answer the question asconcisely as you can. If possible, please provide an answer in one sentence or with some specific entities. Do not provide any explanation.\n\n### Data tables:### \n{context}\n\nOnly give me the answer based on the above information and please provide an answer in one sentence or with some specific entities. Never explain, justify, or add commentary.\n\nQuestion: {input}\nAnswer:"""

PROMPTS["planning_response"]="""
"###Complete the Code Below###\n\nfrom langchain import SerpAPIWrapper\nfrom utils import QA_LLM\nsearch = SerpAPIWrapper()\n\ndef Search(query:str,thought:str):\n    \"\"\"Search relevant information about query based on external Search Engine.\n    Attributes:\n\t\tquery: The question you want to search.\n\t\tthought: The reason why this query is need. \n    \"\"\"\n    if thought is not None:\n        return search.run(query)\n    else:\n        return (\"Please give your thought!\")\n\ndef Get_Answer(query:str,info:str):\n    \"\"\"Get the answer of the query based on the information.\n    Attributes:\n    query: The question you want to search.\n    info: The information relevant to the query.\n    \"\"\"\n    ### Use the QA_LLM model to get the answer.\n    return QA_LLM(query,info)\n\ndef Compare(Original_Query:str,Subquestions:list,Answers:list):\n    \"\"\"Compare the answer of the sub-questions and return the final answer of original query.\n    Attributes:\n    Original_Query: The original question.\n    Subquestions: The list of sub-questions.\n    Answers: The list of answers of the sub-questions.\n    \"\"\"\n    query = Original_Query\n    info = str()\n    for i in range(len(Subquestions)):\n        info += Subquestions[i] + ' : ' + Answers[i] + '\\n'\n    return QA_LLM(query,info)\n\ndef Intersection(Answer1:str,Answer2:str):\n    \"\"\"Find the intersection of two answer sets.\n    Attributes:\n    Answer1: The first answer set.\n    Answer2: The second answer set.\n    \"\"\"\n    List1 = Answer1.split(',')\n    List2 = Answer2.split(',')\n    return str(set(List1) & set(List2))\n\ndef Union(Answer1:str,Answer2:str):\n    \"\"\"Find the union of two answer sets.\n    Attributes:\n    Answer1: The first answer set.\n    Answer2: The second answer set.\n    \"\"\"\n    List1 = Answer1.split(',')\n    List2 = Answer2.split(',')\n    return str(set(List1) | set(List2))\n\ndef Finish_The_Plan(Answer:str):\n    \"\"\"Call this function to finish the plan and return the final answer.\n    Attributes:\n    Answer: The final answer of the original question.\n    \"\"\"\n    return Answer\n\n###################\n# Example 0:\n###################\n\nOriginal_Question: str = \"What is the ethnic group of Booker T. Jones?\"\n### Question Type: One Projection\n### Decompose the original question into sub-questions.\n\nThought1: str = \"An atomic question, no need to decompose. Search directly.\"\nSub_Question_1: str = \"What is the ethnic group of Booker T. Jones?\"\nInfo_1: str = Search(query = Sub_Question_1, thought = Thought1)\nAns_1: str = Get_Answer(query = Sub_Question_1, info = Info_1)\n\nFinal_Answer: str = Finish_The_Plan(Answer = Ans_1)\n\n###################\n# Example 1:\n###################\n\nOriginal_Question: str = \"Who succeeded the first President of Namibia?\"\n### Question Type: Two Projection\n### Decompose the original question into sub-questions.\n\nThought1: str = \"If I want to know who succeeded the first President of Namibia, I need to first know who is the first President of Namibia.\"\nSub_Question_1: str = \"Who is the first President of Namibia?\"\nInfo_1: str = Search(query = Sub_Question_1, thought = Thought1)\nAns_1: str = Get_Answer(query = Sub_Question_1, info = Info_1)\n\nThought2: str = \"After knowing who is the first President of Namibia, I need to know who succeeded him.\"\nSub_Question_2: str = f\"Who succeeded {Ans_1}?\"\nInfo_2: str = Search(query = Sub_Question_2, thought = Thought2)\nAns_2: str = Get_Answer(query = Sub_Question_2, info = Info_2)\n\nFinal_Answer: str = Finish_The_Plan(Answer = Ans_2)\n\n###################\n# Example 2:\n###################\n\nOriginal_Question: str = \"What is the foundational text of Android developer's country?\"\n### Question Type: Three Projection\n### Decompose the original question into sub-questions.\n\nThought1: str = \"If I want to know what is the foundational text of Android developer's country, I need to first know what(who) is the developer of Android.\"\nSub_Question_1: str = \"What(Who) is the developer of Android?\"\nInfo_1: str = Search(query = Sub_Question_1, thought = Thought1)\nAns_1: str = Get_Answer(query = Sub_Question_1, info = Info_1)\n\nThought2: str = \"After knowing what(who) is the developer of Android, I need to the know its country.\"\nSub_Question_2: str = f\"What is the country of {Ans_1}?\"\nInfo_2: str = Search(query = Sub_Question_2, thought = Thought2)\nAns_2: str = Get_Answer(query = Sub_Question_2, info = Info_2)\n\nThought3: str = \"After knowing what is the country of Android developer, I need to know what is the foundational text of country Ans_2.\"\nSub_Question_3: str = f\"What is the foundational text of {Ans_2}?\"\nInfo_3: str = Search(query = Sub_Question_3, thought = Thought3)\nAns_3: str = Get_Answer(query = Sub_Question_3, info = Info_3)\n\nFinal_Answer: str = Finish_The_Plan(Answer = Ans_3)\n\n###################\n# Example 3:\n###################\n\nOriginal_Question: str = \"When was the first establishment that McDonaldization is named after, open in the country Horndean is located?\"\n### Question Type: Entity Replacement\n### Decompose the original question into sub-questions.\n\nThought1: str = \"If I want to know when the first establishment that McDonaldization is named after, open in the country Horndean is located, I need to first know what is McDonaldization named after.\"\nSub_Question_1: str = \"What is McDonaldization named after?\"\nInfo_1: str = Search(query = Sub_Question_1, thought = Thought1)\nAns_1: str = Get_Answer(query = Sub_Question_1, info = Info_1)\n\nThought2: str = \"At the same time, I need to know where the country Horndean is located.\"\nSub_Question_2: str = \"Where is the country Horndean located?\"\nInfo_2: str = Search(query = Sub_Question_2, thought = Thought2)\nAns_2: str = Get_Answer(query = Sub_Question_2, info = Info_2)\n\nThought3: str = \"After knowing what is McDonaldization named after (i.e., Ans_1) and where the country Horndean is located (i.e., Ans_2), I need to know when the first Ans_1 open in the Ans_2.\"\nSub_Question_3: str = f\"When did the first {Ans_1}\\'s open in {Ans_2}?\"\nInfo_3: str = Search(query = Sub_Question_3, thought = Thought3)\nAns_3: str = Get_Answer(query = Sub_Question_3, info = Info_3)\n\nFinal_Answer: str = Finish_The_Plan(Answer = Ans_3)\n\n###################\n# Example 4:\n###################\n\nOriginal_Question: str = \"Which magazine was started first Arthur's Magazine or First for Women?\"\n### Question Type: Compare\n### Decompose the original question into sub-questions.\n\nThought1: str = \"If I want to know which magazine was started first, I need to first know when Arthur's Magazine was started.\"\nSub_Question_1: str = \"When was Arthur's Magazine started?\"\nInfo_1: str = Search(query = Sub_Question_1, thought = Thought1)\nAns_1: str = Get_Answer(query = Sub_Question_1, info = Info_1)\n\nThought2: str = \"At the same time, I need to know when First for Women was started.\"\nSub_Question_2: str = \"When was First for Women started?\"\nInfo_2: str = Search(query = Sub_Question_2, thought = Thought2)\nAns_2: str = Get_Answer(query = Sub_Question_2, info = Info_2)\n\nThought3: str = \"After knowing when Arthur's Magazine was started (i.e., Ans_1) and when First for Women was started, I need to compare the two dates.\"\nAns_3: str = Compare(Original_Query = Original_Question, Subquestions = [Sub_Question_1,Sub_Question_2], Answers = [Ans_1,Ans_2])\n\nFinal_Answer: str = Finish_The_Plan(Answer = Ans_3)\n\n###################\n# Example 5:\n###################\n\nOriginal_Question: str = \"Which areas border with Burlington County and Trumbull County at the same time?\"\n### Question Type: Two Intersection\n### Decompose the original question into sub-questions.\n\nThought1: str = \"If I want to know which areas border with Burlington County and Trumbull County at the same time, I need to first know which areas border with Burlington County.\"\nSub_Question_1: str = \"Which areas border with Burlington County?\"\nInfo_1: str = Search(query = Sub_Question_1, thought = Thought1)\nAns_1: str = Get_Answer(query = Sub_Question_1, info = Info_1)\n\nThought2: str = \"At the same time, I need to know which areas border with Trumbull County.\"\nSub_Question_2: str = \"Which areas border with Trumbull County?\"\nInfo_2: str = Search(query = Sub_Question_2, thought = Thought2)\nAns_2: str = Get_Answer(query = Sub_Question_2, info = Info_2)\n\nThought3: str = \"After knowing which areas border with Burlington County (i.e., Ans_1) and which areas border with Trumbull County (i.e., Ans_2), I need to find the intersection of the two answer sets.\"\nInter_Results1: str = Intersection(Answer1 = Ans_1, Answer2 = Ans_2)\n\nFinal_Answer: str = Finish_The_Plan(Answer = Inter_Results1)\n\n\n###################\n# Example 6:\n###################\n\nOriginal_Question: str = \"What are the same genre shared between Alice in Wonderland, Blues Brothers 2000 and Pinocchio?\"\n### Question Type: Three Intersection\n### Decompose the original question into sub-questions.\n\nThought1: str = \"If I want to know what are the same genre shared between Alice in Wonderland, Blues Brothers 2000 and Pinocchio, I need to first know what is the genre of Alice in Wonderland.\"\nSub_Question_1: str = \"What is the genre of Alice in Wonderland?\"\nInfo_1: str = Search(query = Sub_Question_1, thought = Thought1)\nAns_1: str = Get_Answer(query = Sub_Question_1, info = Info_1)\n\nThought2: str = \"At the same time, I need to know what is the genre of Blues Brothers 2000.\"\nSub_Question_2: str = \"What is the genre of Blues Brothers 2000?\"\nInfo_2: str = Search(query = Sub_Question_2, thought = Thought2)\nAns_2: str = Get_Answer(query = Sub_Question_2, info = Info_2)\n\nThought3: str = \"At the same time, I need to know what is the genre of Pinocchio.\"\nSub_Question_3: str = \"What is the genre of Pinocchio?\"\nInfo_3: str = Search(query = Sub_Question_3, thought = Thought3)\nAns_3: str = Get_Answer(query = Sub_Question_3, info = Info_3)\n\nThought4: str = \"After knowing what is the genre of Alice in Wonderland (i.e., Ans_1), what is the genre of Blues Brothers 2000 (i.e., Ans_2) and what is the genre of Pinocchio (i.e., Ans_3), I need to find the intersection of the three answer sets.\"\nInter_Results1: str = Intersection(Answer1 = Ans_1, Answer2 = Ans_2)\nInter_Results2: str = Intersection(Answer1 = Inter_Results1, Answer2 = Ans_3)\n\nFinal_Answer: str = Finish_The_Plan(Answer = Inter_Results2)\n\n###################\n# Example 7:\n###################\n\nOriginal_Question: str = \"Who are all the cast members from 'Wuthering Heights' combined with the cast members from 'Traffic'?\"\n### Question Type: Two Union\n### Decompose the original question into sub-questions.\n\nThought1: str = \"If I want to know who are all the cast members from 'Wuthering Heights' combined with the cast members from 'Traffic', I need to first know who are the cast members from 'Wuthering Heights'.\"\nSub_Question_1: str = \"Who are the cast members from 'Wuthering Heights'?\"\nInfo_1: str = Search(query = Sub_Question_1, thought = Thought1)\nAns_1: str = Get_Answer(query = Sub_Question_1, info = Info_1)\n\nThought2: str = \"At the same time, I need to know who are the cast members from 'Traffic'.\"\nSub_Question_2: str = \"Who are the cast members from 'Traffic'?\"\nInfo_2: str = Search(query = Sub_Question_2, thought = Thought2)\nAns_2: str = Get_Answer(query = Sub_Question_2, info = Info_2)\n\nThought3: str = \"After knowing who are the cast members from 'Wuthering Heights' (i.e., Ans_1) and who are the cast members from 'Traffic' (i.e., Ans_2), I need to find the union of the two answer sets.\"\nUnion_Results1: str = Union(Answer1 = Ans_1, Answer2 = Ans_2)\n\nFinal_Answer: str = Finish_The_Plan(Answer = Union_Results1)\n\n###################\n# Example 8:\n###################\n\nOriginal_Question: str = \"Which regions border Drake Bell's birthplace and Santa Ana at the same time?\"\n### Question Type: Projection then Two Intersection\n### Decompose the original question into sub-questions.\n\nThought1: str = \"If I want to know which regions border Drake Bell's birthplace and Santa Ana at the same time, I need to first know where is Drake Bell's birthplace.\"\nSub_Question_1: str = \"Where is Drake Bell's birthplace?\"\nInfo_1: str = Search(query = Sub_Question_1, thought = Thought1)\nAns_1: str = Get_Answer(query = Sub_Question_1, info = Info_1)\n\nThought2: str = \"Then I need to know which regions border with Drake Bell's birthplace.(i.e., Ans_1)\"\nSub_Question_2: str = f\"Which regions border with {Ans_1}?\"\nInfo_2: str = Search(query = Sub_Question_2, thought = Thought2)\nAns_2: str = Get_Answer(query = Sub_Question_2, info = Info_2)\n\nThought3: str = \"At the same time, I need to know which regions border with Santa Ana.\"\nSub_Question_3: str = \"Which regions border with Santa Ana?\"\nInfo_3: str = Search(query = Sub_Question_3, thought = Thought3)\nAns_3: str = Get_Answer(query = Sub_Question_3, info = Info_3)\n\nThought4: str = \"After knowing which regions border with Drake Bell's birthplace (i.e., Ans_2) and which regions border with Santa Ana (i.e., Ans_3), I need to find the intersection of the two answer sets.\"\nInter_Results1: str = Intersection(Answer1 = Ans_2, Answer2 = Ans_3)\n\nFinal_Answer: str = Finish_The_Plan(Answer = Inter_Results1)\n\n###################\n# Example 9:\n###################\n\nOriginal_Question: str = \"What are the political party of people who are cast members of both The Blues Brothers and Going My Way?\"\n### Question Type: Two Intersection then Projection\n### Decompose the original question into sub-questions.\n\nThought1: str = \"If I want to know what are the political party of people who are cast members of both The Blues Brothers and Going My Way, I need to first know who are the cast members of The Blues Brothers.\"\nSub_Question_1: str = \"Who are the cast members of The Blues Brothers?\"\nInfo_1: str = Search(query = Sub_Question_1, thought = Thought1)\nAns_1: str = Get_Answer(query = Sub_Question_1, info = Info_1)\n\nThought2: str = \"At the same time, I need to know who are the cast members of Going My Way.\"\nSub_Question_2: str = \"Who are the cast members of Going My Way?\"\nInfo_2: str = Search(query = Sub_Question_2, thought = Thought2)\nAns_2: str = Get_Answer(query = Sub_Question_2, info = Info_2)\n\nThought3: str = \"After knowing who are the cast members of The Blues Brothers (i.e., Ans_1) and who are the cast members of Going My Way (i.e., Ans_2), I need to find the intersection of the two answer sets.\"\nInter_Results1: str = Intersection(Answer1 = Ans_1, Answer2 = Ans_2)\n\nThought4: str = \"Then I need to know the political party of the people in Inter_Results1.\"\nSub_Question_3: str = f\"What are the political party of people in {Inter_Results1}?\"\nInfo_3: str = Search(query = Sub_Question_3, thought = Thought4)\nAns_3: str = Get_Answer(query = Sub_Question_3, info = Info_3)\n\nFinal_Answer: str = Finish_The_Plan(Answer = Ans_3)\n\n###################\n# Your turn! Just complete the code below and do not return other things.\n###################\n\nOriginal_Question: str = \"{_original_query_}\"\n"
"""

PROMPTS["training_summary"]='''
You are an expert in filtering triples and summarizing text. Your task is to complete the following tasks based on the given question, data table, and the answer to the question:
1. For the relation triples in the given data table, filter out only the top 15 triples that are most relevant to the question.
2. For each filtered relation triple, re-score the weight based on the relevance of the relation triple to the question. The weight value ranges from 0.0 to 10.0.
4. Summarize the text in the data table based on the given text (Source), and output a summary that is relevant to the question without copying the original text verbatim and without fabricating nonexistent information.
5. Output should be in the same format as the given data tables and the output should include two sections: "Relationships" and "Sources".
6. When finished, output <|end|>

Here are two examples (Note the format of "Output"):

---Example 1---

Question:
who was the captain of the mayflower which brought the pilgrims to plymouth?

Data tables:
'\n-----Relationships-----\n```csv\nid,source entity,relationship between the source entity and the target entity,target entity,weight,rank\r\n0,"""EDWARD WINSLOW""","""prominent in the church and wrote important documents about""","""PLYMOUTH COLONY""",9.0,14\r\n1,"""GEORGE SOULE""","""signer of the Mayflower Compact and helped establish""","""PLYMOUTH COLONY""",9.0,13\r\n2,"""JOHN ALDEN SR.""","""stayed after voyage""","""PLYMOUTH COLONY""",8.0,12\r\n3,"""EDWARD WINSLOW""","""prominent in the same church and colony""<SEP>""was part of Edward Winslow\'s household as a manservant or apprentice""","""GEORGE SOULE""",8.0,11\r\n4,"""GEORGE SOULE""","""prominent in the same church and colony""<SEP>""was part of Edward Winslow\'s household as a manservant or apprentice""","""EDWARD WINSLOW""",8.0,11\r\n5,"""JOHN CARVER (DECEASED GOVERNOR)""","""became the first Governor after voyage""","""PLYMOUTH COLONY""",9.0,10\r\n6,"""WILLIAM BRADFORD (GOVERNOR)""","""prominent leader and founder of the colony""","""PLYMOUTH COLONY""",10.0,9\r\n7,"""STEPHEN HOPKINS (VETERAN OF FAILED COLONIAL VENTURE)""","""part of the group merging ships with Leiden Leaders""","""PLYMOUTH COLONY""",9.0,9\r\n8,"""DOROTHY CARVER (WIFE OF JOHN CARVER)""","""died in 1633 at anchor""","""PLYMOUTH COLONY""",9.0,9\r\n9,"""WILLIAM BREWSTER\'S SCHEME""","""had a long-term effect on""","""PLYMOUTH COLONY""",8.0,9\r\n10,"""EDWARD WINSLOW""","""traveling companion on the Mayflower with""","""DOROTHY (WIFE)""",8.0,8\r\n11,"""EDWARD WINSLOW""","""became after John Carver\'s death""","""COLONIAL GOVERNOR""",10.0,7\r\n12,"""EDWARD WINSLOW""","""took over as Governor after death""","""JOHN CARVER""",10.0,7\r\n13,"""EDWARD WINSLOW""","""came to as a Separatist""","""LEIDEN, HOLLAND""",9.0,7\r\n14,"""CHRISTOPHER MARTIN (STRANGER AND MERCHANT ADVENTURER REPRESENTATIVE)""","""came on the ship with family and servants""","""MAYFLOWER""",8.0,7\r\n15,"""CARVER AND HIS WIFE KATHERINE (FAMILY)""","""boarded with servants, child, and served as governor during crossing""","""MAYFLOWER""",9.0,6\r\n16,"""GEORGE SOULE""","""traveling on the Mayflower with his indentured servants and fellow colonists""","""MAYFLOWER PASSENGER""",8.0,6\r\n17,"""MAYFLOWER""","""departed on this date with passengers and crew""","""SEPTEMBER 6, 1620 DEPARTURE DATE""",8.0,6\r\n18,"""MAYFLOWER""","""embarked about""","""65 PASSENGERS""",8.0,6\r\n19,"""MASTER LEAVER (IDENTIFIED AS POSSIBLE PRINCIPAL OFFICER)""","""possibly a principal officer of the ship""","""MAYFLOWER""",7.0,6\r\n20,"""GEORGE SOULE""","""traveling companion who died in the first winter with""","""ELIAS STORY""",7.0,6\r\n21,"""GEORGE SOULE""","""traveling companion who died in the first winter with""","""ELLEN MORE""",6.0,6\r\n22,"""JOHN ALDEN SR.""","""married to""","""PRISCILLA MULLINS""",10.0,5\r\n23,"""PRISCILLA MULLINS (PASSENGER)""","""married John Alden Sr.""","""JOHN ALDEN SR.""",9.0,5\r\n24,"""JOHN ALDEN SR.""","""signed the compact as a crew member""","""MAYFLOWER COMPACT (SIGNATORY)""",7.0,5\r\n25,"""CHRISTOPHER MARTIN (STRANGER AND MERCHANT ADVENTURER REPRESENTATIVE)""","""representative for""","""NON-RELIGIOUS PASSENGERS OF MAYFLOWER""",8.0,3\r\n26,"""JOHN CARVER (DECEASED GOVERNOR)""","""initial elected governor for Atlantic crossing""","""MAYFLOWER COMPACT SIGNATORY""",8.0,3\r\n27,"""DOROTHY (WIFE)""","""drowned while ship was anchored here""","""CAPE COD HARBOR""",8.0,3\r\n28,"""STEERAGE ROOM""","""probably also housed""","""SHIP\'S COMPASS""",6.0,3\r\n29,"""STEERAGE ROOM""","""housed""","""WHIPSTAFF (TILLER EXTENSION)""",5.0,3\r\n30,"""THOMAS WESTON AND MERCHANT ADVENTURERS, LONDON BUSINESSMEN""","""provided funding for the voyage""","""MAYFLOWER VOYAGE FUNDING""",9.0,2\r\n31,"""EDWARD WINSLOW, WILLIAM BREWSTER, ISAAC ALLERTON (LEIDEN LEADERS)""","""leaders in the group merging ships""","""PILGRIMS""",9.0,2\r\n32,"""WILLIAM BRADFORD, EDWARD WINSLOW, WILLIAM BREWSTER, ISAAC ALLERTON (LEIDEN LEADERS)""","""part of the group after merging ships""","""LEIDEN CONGREGATION""",9.0,2\r\n33,"""CARVER (WEALTHY INVESTOR)""","""invested personal fortune in the voyage""","""MAYFLOWER VOYAGE INVESTMENT""",9.0,2\r\n34,"""GILES HEALE (SURGEON)""","""witness to death-bed will""","""WILLIAM MULLINS (PASSENGER)""",9.0,2\r\n35,"""CARVER (VESSEL ORGANIZER)""","""organized voyage and negotiations with funders""","""WESTON AND MERCHANT ADVENTURERS""",8.0,2\r\n36,"""WILLIAM MULLINS (FAMILY)""","""perished in first winter with family""","""ALL PASSENGERS OF MAYFLOWER""",8.0,2\r\n37,"""FORECASTLE SPACE""","""space for various roles""","""SHIP\'S COOK, SHIP\'S SURGEON, AND SHIP\'S OFFICERS""",8.0,2\r\n38,"""MASTER CHRISTOPHER JONES (CABIN)""","""measuring""","""TEN BY SEVEN FEET""",7.0,2\r\n39,"""CAPSTAN""","""used to pull in ropes or cables""","""VERTICAL AXLE""",4.0,2\r\n\n```\n-----Sources-----\n```csv\nid,content\r\n0,"to the Separatist Church in Nottinghamshire England who came to Leiden, Holland about 1608 and became prominent in the church there. He came on the ""Mayflower"" with his wife Dorothy, leaving a young son in Leiden; Dorothy drowned while the ship was at anchor in Cape Cod Harbor. He became colony Governor after the death of John Carver, and was prominent in the Plymouth Church. His writings of early Plymouth Colony are important historic documents. Edward Winslow - A gentleman from a well-off family who was prominent in the Separatist church in Leiden and involved with Brewster in printing anti-Anglican\nPassage 10: George Soule (Mayflower passenger) George Soule (c. 1601 – between 20 September 1677 and 22 January 1679) was a colonist who was one of the indentured servants on the ""Mayflower"" and helped establish Plymouth Colony in 1620. He was one of the signers of the Mayflower Compact. It is known that George came on the ""Mayflower"" and was credited to the household of Edward Winslow as a manservant or apprentice, along with Elias Story and a little girl Ellen More, who both died in the first winter. What is not generally understood is the long-term effect of William Brewster\'s scheme"\r\n1,"Passage 1: Aft on the main deck in the stern was the cabin for Master Christopher Jones, measuring about ten by seven feet (3 m × 2.1 m).  Forward of that was the steerage room, which housed a whipstaff (tiller extension) for sailing control; not a wheel, as in later ships. Also here was the ship\'s compass and probably also berths for the ship\'s officers. Forward of the steerage room was the capstan, a vertical axle used to pull in ropes or cables. Far forward on the main deck, just aft of the bow, was the forecastle space where the ship\'s cook\nPassage 2: the ship\'s surgeon, a young man just out of apprenticeship as a London Barber-Surgeon by the name of Giles Heale. His name appears as a witness to the death-bed will of William Mullins in February 1621. Another person that Bradford also did not mention who is recorded as possibly being a principal officer of the Mayflower due to his title, is a man identified only as ""Master"" Leaver. He is recorded in Mourt\'s Relation (1622) as rescuing Pilgrims lost in a forest in January 1621. ""Mayflower"" embarked about sixty-five passengers in London about the middle of July 1620, proceeded to\nPassage 3: Carver and his wife Katherine boarded ""Mayflower"" with five servants and seven year-old Jasper More, one of the four children of the More family who were sent in the care of the Pilgrims. Carver seems to have been elected governor of the ""Mayflower"" for the duration of the Atlantic crossing. The ""Mayflower"" anchored off Cape Cod in November, 1620, and the Mayflower Compact was signed aboard ship on November 11; it became the first governing document for Plymouth Colony. Carver may have been the author of the Compact, and was definitely its first signer. He was subsequently chosen to be\nPassage 4: John Alden Capt. John Alden Sr. ( 15981687) was a crew member on the historic 1620 voyage of the Pilgrim ship ""Mayflower"". Rather than return to England with the ship, he stayed at what became Plymouth Colony. He was hired in Southampton, England, as the ship\'s cooper, responsible for maintaining the ship\'s barrels. He was a signatory to the Mayflower Compact. He married fellow ""Mayflower"" passenger Priscilla Mullins, whose entire family perished in the first winter. He served in a number of important government positions such as Assistant Governor, Duxbury Deputy to the General Court of Plymouth, Captain Myles Standish\'s\nPassage 5: supremacy of the king and the Church of England. To fund the ""Mayflower"" voyage, the Leiden congregation turned to Thomas Weston and the Merchant Adventurers, London businessmen interested in supporting the voyage in hopes of profit. Carver had the task of organizing the voyage and negotiating funding with Weston and the Adventurers, along with Cushman as the chief agent. Carver was in Southampton in June 1620 purchasing supplies for the ""Mayflower"" voyage, along with Christopher Martin. Carver was very wealthy and provided much of his personal fortune to make investment in the joint-stock company and in the Mayflower voyage itself.\nPassage 6: a week, in an effect to increase profits, without such as due time for religious activities. The Pilgrims balked at this and refused to agree to the new terms. William Mullins played a part in these deliberations, probably because he had a large investment and needed to ensure a satisfactory return on it, as an Adventurers member. And although Robert Cushman, who had been the Leiden agent for ""Mayflower"" voyage preparations, came to Plymouth in November 1621 to try to settle the rift between the Pilgrims and the Adventurers, it was never resolved. Eventually the Pilgrims bought out the Adventurers\nPassage 7: had been designated by the Merchant Adventurers to act as shipboard governor during the trans-Atlantic trip; and Stephen Hopkins, a veteran of a failed colonial venture that may have inspired Shakespeare\'s ""The Tempest"". The group who later became the Leiden Leaders after the merging of ships included John Carver, William Bradford, Edward Winslow, William Brewster, and Isaac Allerton. The ""Mayflower"" departed Plymouth, England on September 6, 1620 with 102 passengers and about 30 crew members in the small, 106 foot-long ship. The seas were not severe during the first month in the Atlantic but, by the second month, the ship\nPassage 8: Plymouth Colony. He left his family in Leiden and came on the ""Mayflower"" with only young servant William Butten, who died at sea a few days before reaching Cape Cod. He was the largely self-taught physician and surgeon of the colony and died in 1633 of an infectious fever that killed many that year. Christopher Martin - He was a prosperous leader of those non-religious persons known as ""Strangers"" on the ""Mayflower"", as well as a representative of the Merchant Adventurer investment group. He came on the ship with his wife and two servants, one of whom was his step-son\nPassage 9: to the Separatist Church in Nottinghamshire England who came to Leiden, Holland about 1608 and became prominent in the church there. He came on the ""Mayflower"" with his wife Dorothy, leaving a young son in Leiden; Dorothy drowned while the ship was at anchor in Cape Cod Harbor. He became colony Governor after the death of John Carver, and was prominent in the Plymouth Church. His writings of early Plymouth Colony are important historic documents. Edward Winslow - A gentleman from a"\r\n\n```\n'

The answer to this question:
Master Christopher Jones, Christopher Jones

Output:
'\n-----Relationships-----\n```csv\nid,source entity,relationship between the source entity and the target entity,target entity,weight,rank\r\n14,"""CHRISTOPHER MARTIN (STRANGER AND MERCHANT ADVENTURER REPRESENTATIVE)""","""came on the ship with family and servants""","""MAYFLOWER""",3.0,7\r\n15,"""CARVER AND HIS WIFE KATHERINE (FAMILY)""","""boarded with servants, child, and served as governor during crossing""","""MAYFLOWER""",3.0,6\r\n17,"""MAYFLOWER""","""departed on this date with passengers and crew""","""SEPTEMBER 6, 1620 DEPARTURE DATE""",5.0,6\r\n18,"""MAYFLOWER""","""embarked about""","""65 PASSENGERS""",5.0,6\r\n19,"""MASTER LEAVER (IDENTIFIED AS POSSIBLE PRINCIPAL OFFICER)""","""possibly a principal officer of the ship""","""MAYFLOWER""",1.0,6\r\n24,"""JOHN ALDEN SR.""","""signed the compact as a crew member""","""MAYFLOWER COMPACT (SIGNATORY)""",3.0,5\r\n26,"""JOHN CARVER (DECEASED GOVERNOR)""","""initial elected governor for Atlantic crossing""","""MAYFLOWER COMPACT SIGNATORY""",3.0,3\r\n30,"""THOMAS WESTON AND MERCHANT ADVENTURERS, LONDON BUSINESSMEN""","""provided funding for the voyage""","""MAYFLOWER VOYAGE FUNDING""",2.0,2\r\n31,"""EDWARD WINSLOW, WILLIAM BREWSTER, ISAAC ALLERTON (LEIDEN LEADERS)""","""leaders in the group merging ships""","""PILGRIMS""",2.0,2\r\n35,"""CARVER (VESSEL ORGANIZER)""","""organized voyage and negotiations with funders""","""WESTON AND MERCHANT ADVENTURERS""",3.0,2\r\n38,"""MASTER CHRISTOPHER JONES (CABIN)""","""measuring""","""TEN BY SEVEN FEET""",10.0,2\r\n5,"""JOHN CARVER (DECEASED GOVERNOR)""","""became the first Governor after voyage""","""PLYMOUTH COLONY""",2.0,10\r\n6,"""WILLIAM BRADFORD (GOVERNOR)""","""prominent leader and founder of the colony""","""PLYMOUTH COLONY""",2.0,9\r\n7,"""STEPHEN HOPKINS (VETERAN OF FAILED COLONIAL VENTURE)""","""part of the group merging ships with Leiden Leaders""","""PLYMOUTH COLONY""",2.0,9\r\n12,"""EDWARD WINSLOW""","""took over as Governor after death""","""JOHN CARVER""",2.0,7\r\n\n```\n-----Sources-----\n```csv\nid,content\r\n0,"Master Christopher Jones was the commander of the Mayflower, with a private cabin located at the stern of the main deck, measuring approximately 10 by 7 feet. Adjacent to his cabin was the steerage room, housing essential navigation equipment like the ship’s compass and a whipstaff for steering. The area also included the capstan, used for hauling ropes, and forward sections like the forecastle, where the cook worked. His role as the ship’s master placed him in charge of navigation and overall operations during the 1620 voyage to Plymouth Colony. No further personal details or direct involvement in colonial governance are noted in the provided text."\r\n\n```\n'<|end|>

---Example 2---

Question:
Where was the director of film French Heels born?

Data tables:
'\n-----Relationships-----\n```csv\nid,source entity,relationship between the source entity and the target entity,target entity,weight,rank\r\n0,"""BRIAN PATRICK KENNEDY""","""is currently the director of""","""PEABODY ESSEX MUSEUM""",10.0,5\r\n1,"""FRENCH HEELS""","""directed by""","""EDWIN L. HOLLYWOOD""",8.0,5\r\n2,"""OLAV AARAAS""","""has been the director since 2001""","""NORWEGIAN MUSEUM OF CULTURAL HISTORY""",8.0,5\r\n3,"""OLAV AARAAS""","""is a ""","""NORWEGIAN HISTORIAN AND MUSEUM DIRECTOR""",8.0,5\r\n4,"""BRIAN PATRICK KENNEDY""","""was the director from 2010 to 2019""","""TOLEDO MUSEUM OF ART""",8.0,5\r\n5,"""OLAV AARAAS""","""was the director from 1993 to 2010""","""MAIHAUGEN""",7.0,5\r\n6,"""BRIAN PATRICK KENNEDY""","""was the director from 2005 to 2010""","""HOOD MUSEUM OF ART""",7.0,5\r\n7,"""OLAV AARAAS""","""was the director from 1982 to 1993""","""SOGN FOLK MUSEUM""",6.0,5\r\n8,"""BRIAN PATRICK KENNEDY""","""was the director in Canberra from 1997-2004""","""NATIONAL GALLERY OF AUSTRALIA""",6.0,5\r\n9,"""EDWIN L. HOLLYWOOD""","""directed by him ""","""1922 AMERICAN SILENT ROMANTIC COMEDY FILM \'FRENCH HEELS\'""",8.0,4\r\n10,"""EDWIN L. HOLLYWOOD""","""in charge of starring Harry Morey""","""VITAGRAPH\'S FILM UNIT""",8.0,4\r\n11,"""S.N. MATHUR""","""was the Director from September 1975 to February 1980""","""INDIAN INTELLIGENCE BUREAU""",6.0,4\r\n12,"""S.N. MATHUR""","""was between certain years""","""DIRECTOR OF THE INDIAN INTELLIGENCE BUREAU""",6.0,4\r\n13,"""S.N. MATHUR""","""was Director General of in Punjab""<SEP>""was the Director General of between certain years""","""PUNJAB POLICE FORCE""",5.0,4\r\n14,"""DANA BLANKSTEIN- COHEN""","""is the director of""","""ISRAELI ACADEMY OF FILM AND TELEVISION""",10.0,3\r\n15,"""DANA BLANKSTEIN- COHEN""","""is the director of ""","""ISRAELI ACADEMY OF FILM AND TELEVISION DIRECTOR""",8.0,3\r\n16,"""JESSE EDWARD HOBSON""","""was the director from 1947 to 1955""","""SRI INTERNATIONAL""",8.0,3\r\n17,"""PETER LEVIN""","""works in as a director""","""FILM, TELEVISION AND THEATRE INDUSTRY""",7.0,3\r\n18,"""JASON MOORE""","""works in as a director ""","""FILM, THEATRE AND TELEVISION INDUSTRY DIRECTOR""",7.0,3\r\n19,"""JASON MOORE""","""works in as a director""","""FILM, THEATRE AND TELEVISION INDUSTRY""",7.0,3\r\n20,"""PETER LEVIN""","""works in as a director""","""AMERICAN FILM, TELEVISION AND THEATRE INDUSTRY""",7.0,3\r\n21,"""IAN BARRY""","""works in as a director""","""FILM AND TV INDUSTRY""",7.0,3\r\n22,"""IAN BARRY""","""works in as a director ""","""AUSTRALIAN FILM AND TV INDUSTRY""",7.0,3\r\n23,"""FRENCH HEELS""","""starred in""","""IRENE CASTLE""",6.0,3\r\n24,"""JESSE EDWARD HOBSON""","""was the director prior to SRI International ""<SEP>""was the director prior to SRI International""","""ARMOUR RESEARCH FOUNDATION""",6.0,3\r\n\n```\n-----Sources-----\n```csv\nid,content\r\n0,"Passage 1: Brian Patrick Kennedy( born 5 November 1961) is an Irish- born art museum director who has worked in Ireland and Australia, and now lives and works in the United States.\nHe is currently the director of the Peabody Essex Museum.\nHe was the director of the Toledo Museum of Art in Ohio from 2010 to 2019.\nHe was the director of the Hood Museum of Art from 2005 to 2010, and the National Gallery of Australia( Canberra) from 1997- 2004.\nPassage 2: Olav Aaraas( born 10 July 1950) is a Norwegian historian and museum director.\nHe was born in Fredrikstad.\nFrom 1982 to 1993 he was the director of Sogn Folk Museum, from 1993 to 2010 he was the director of Maihaugen and from 2001 he has been the director of the Norwegian Museum of Cultural History.\nIn 2010 he was decorated with the Royal Norwegian Order of St. Olav.\nPassage 3: S.N. Mathur was the Director of the Indian Intelligence Bureau between September 1975 and February 1980.\nHe was also the Director General of Police in Punjab.\nPassage 4: Peter Levin is an American director of film, television and theatre.\nPassage 5: French Heels is a lost 1922 American silent romantic comedy film directed by Edwin L. Hollywood and starring Irene Castle.\nBased on short story"" Knots and Windshakes"" by Clarence Budington Kelland which appeared in"" Everybody\'s Magazine"", it was distributed by W. W. Hodkinson.\nPassage 6: Dana Blankstein- Cohen( born March 3, 1981) is the director of the Israeli Academy of Film and Television.\nShe is a film director, and an Israeli culture entrepreneur.\nPassage 7: Ian Barry is an Australian director of film and TV.\nPassage 8: Edwin L. Hollywood (October 9, 1892 – May 15, 1958) was an American actor and film director.\nHe was born in New York City.\nHollywood was in charge of Vitagraph\'s film unit that starred Harry Morey.\nHe died in Glendale, California.\nPassage 9: Jason Moore( born October 22, 1970) is an American director of film, theatre and television.\nPassage 10: Jesse Edward Hobson( May 2, 1911 – November 5, 1970) was the director of SRI International from 1947 to 1955.\nPrior to SRI, he was the director of the Armour Research Foundation."\r\n\n```\n'

The answer to this question:
New York

Output:
'\n-----Relationships-----\n```csv\nid,source entity,relationship between the source entity and the target entity,target entity,weight,rank\r\n1,"""FRENCH HEELS""","""directed by""","""EDWIN L. HOLLYWOOD""",10.0,5\r\n9,"""EDWIN L. HOLLYWOOD""","""directed by him ""","""1922 AMERICAN SILENT ROMANTIC COMEDY FILM \'FRENCH HEELS\'""",9.0,4\r\n10,"""EDWIN L. HOLLYWOOD""","""in charge of starring Harry Morey""","""VITAGRAPH\'S FILM UNIT""",7.0,4\r\n8,"""EDWIN L. HOLLYWOOD""","""born in""","""NEW YORK CITY""",10.0,5\r\n23,"""FRENCH HEELS""","""starred in""","""IRENE CASTLE""",5.0,3\r\n0,"""BRIAN PATRICK KENNEDY""","""is currently the director of""","""PEABODY ESSEX MUSEUM""",2.0,5\r\n2,"""OLAV AARAAS""","""has been the director since 2001""","""NORWEGIAN MUSEUM OF CULTURAL HISTORY""",2.0,5\r\n4,"""BRIAN PATRICK KENNEDY""","""was the director from 2010 to 2019""","""TOLEDO MUSEUM OF ART""",2.0,5\r\n5,"""OLAV AARAAS""","""was the director from 1993 to 2010""","""MAIHAUGEN""",2.0,5\r\n6,"""BRIAN PATRICK KENNEDY""","""was the director from 2005 to 2010""","""HOOD MUSEUM OF ART""",2.0,5\r\n14,"""DANA BLANKSTEIN- COHEN""","""is the director of""","""ISRAELI ACADEMY OF FILM AND TELEVISION""",2.0,3\r\n16,"""JESSE EDWARD HOBSON""","""was the director from 1947 to 1955""","""SRI INTERNATIONAL""",2.0,3\r\n17,"""PETER LEVIN""","""works in as a director""","""FILM, TELEVISION AND THEATRE INDUSTRY""",2.0,3\r\n21,"""IAN BARRY""","""works in as a director""","""FILM AND TV INDUSTRY""",2.0,3\r\n\n```\n-----Sources-----\n```csv\nid,content\r\n0,"Edwin L. Hollywood, director of the 1922 silent film *French Heels*, was born in New York City. He began his career as an actor before transitioning to directing, notably leading Vitagraph\'s film unit featuring Harry Morey. The romantic comedy starred Irene Castle and was based on a Clarence Budington Kelland short story. No other biographical details about Hollywood’s early life or education are provided in the text."\r\n\n```\n'<|end|>

---Your turn---
Only give me the output. Never explain, justify, or add commentary!!!

Question:
{input_question}

Data tables:
'{input_data_table}'

The answer to this question:
{input_answer}

Output:
'''


PROMPTS["summary_response"]='''
You are an expert in filtering triples and summarizing text. Your task is to complete the following tasks based on the given question and data tables:
1. For the relation triples in the given data table, filter out only the top 15 triples that are most relevant to the question.
2. For each filtered relation triple, re-score the weight based on the relevance of the relation triple to the question. The weight value ranges from 0.0 to 10.0.
4. Summarize the text in the data table based on the given text (Source), and output a summary that is relevant to the question without copying the original text verbatim and without fabricating nonexistent information.
5. Output should be in the same format as the given data tables and the output should include two sections: "Relationships" and "Sources".
6. When finished, output <|end|>

Here are two examples (Note the format of "Output"):

---Example 1---

Question:
who was the captain of the mayflower which brought the pilgrims to plymouth?

Data tables:
'\n-----Relationships-----\n```csv\nid,source entity,relationship between the source entity and the target entity,target entity,weight,rank\r\n0,"""EDWARD WINSLOW""","""prominent in the church and wrote important documents about""","""PLYMOUTH COLONY""",9.0,14\r\n1,"""GEORGE SOULE""","""signer of the Mayflower Compact and helped establish""","""PLYMOUTH COLONY""",9.0,13\r\n2,"""JOHN ALDEN SR.""","""stayed after voyage""","""PLYMOUTH COLONY""",8.0,12\r\n3,"""EDWARD WINSLOW""","""prominent in the same church and colony""<SEP>""was part of Edward Winslow\'s household as a manservant or apprentice""","""GEORGE SOULE""",8.0,11\r\n4,"""GEORGE SOULE""","""prominent in the same church and colony""<SEP>""was part of Edward Winslow\'s household as a manservant or apprentice""","""EDWARD WINSLOW""",8.0,11\r\n5,"""JOHN CARVER (DECEASED GOVERNOR)""","""became the first Governor after voyage""","""PLYMOUTH COLONY""",9.0,10\r\n6,"""WILLIAM BRADFORD (GOVERNOR)""","""prominent leader and founder of the colony""","""PLYMOUTH COLONY""",10.0,9\r\n7,"""STEPHEN HOPKINS (VETERAN OF FAILED COLONIAL VENTURE)""","""part of the group merging ships with Leiden Leaders""","""PLYMOUTH COLONY""",9.0,9\r\n8,"""DOROTHY CARVER (WIFE OF JOHN CARVER)""","""died in 1633 at anchor""","""PLYMOUTH COLONY""",9.0,9\r\n9,"""WILLIAM BREWSTER\'S SCHEME""","""had a long-term effect on""","""PLYMOUTH COLONY""",8.0,9\r\n10,"""EDWARD WINSLOW""","""traveling companion on the Mayflower with""","""DOROTHY (WIFE)""",8.0,8\r\n11,"""EDWARD WINSLOW""","""became after John Carver\'s death""","""COLONIAL GOVERNOR""",10.0,7\r\n12,"""EDWARD WINSLOW""","""took over as Governor after death""","""JOHN CARVER""",10.0,7\r\n13,"""EDWARD WINSLOW""","""came to as a Separatist""","""LEIDEN, HOLLAND""",9.0,7\r\n14,"""CHRISTOPHER MARTIN (STRANGER AND MERCHANT ADVENTURER REPRESENTATIVE)""","""came on the ship with family and servants""","""MAYFLOWER""",8.0,7\r\n15,"""CARVER AND HIS WIFE KATHERINE (FAMILY)""","""boarded with servants, child, and served as governor during crossing""","""MAYFLOWER""",9.0,6\r\n16,"""GEORGE SOULE""","""traveling on the Mayflower with his indentured servants and fellow colonists""","""MAYFLOWER PASSENGER""",8.0,6\r\n17,"""MAYFLOWER""","""departed on this date with passengers and crew""","""SEPTEMBER 6, 1620 DEPARTURE DATE""",8.0,6\r\n18,"""MAYFLOWER""","""embarked about""","""65 PASSENGERS""",8.0,6\r\n19,"""MASTER LEAVER (IDENTIFIED AS POSSIBLE PRINCIPAL OFFICER)""","""possibly a principal officer of the ship""","""MAYFLOWER""",7.0,6\r\n20,"""GEORGE SOULE""","""traveling companion who died in the first winter with""","""ELIAS STORY""",7.0,6\r\n21,"""GEORGE SOULE""","""traveling companion who died in the first winter with""","""ELLEN MORE""",6.0,6\r\n22,"""JOHN ALDEN SR.""","""married to""","""PRISCILLA MULLINS""",10.0,5\r\n23,"""PRISCILLA MULLINS (PASSENGER)""","""married John Alden Sr.""","""JOHN ALDEN SR.""",9.0,5\r\n24,"""JOHN ALDEN SR.""","""signed the compact as a crew member""","""MAYFLOWER COMPACT (SIGNATORY)""",7.0,5\r\n25,"""CHRISTOPHER MARTIN (STRANGER AND MERCHANT ADVENTURER REPRESENTATIVE)""","""representative for""","""NON-RELIGIOUS PASSENGERS OF MAYFLOWER""",8.0,3\r\n26,"""JOHN CARVER (DECEASED GOVERNOR)""","""initial elected governor for Atlantic crossing""","""MAYFLOWER COMPACT SIGNATORY""",8.0,3\r\n27,"""DOROTHY (WIFE)""","""drowned while ship was anchored here""","""CAPE COD HARBOR""",8.0,3\r\n28,"""STEERAGE ROOM""","""probably also housed""","""SHIP\'S COMPASS""",6.0,3\r\n29,"""STEERAGE ROOM""","""housed""","""WHIPSTAFF (TILLER EXTENSION)""",5.0,3\r\n30,"""THOMAS WESTON AND MERCHANT ADVENTURERS, LONDON BUSINESSMEN""","""provided funding for the voyage""","""MAYFLOWER VOYAGE FUNDING""",9.0,2\r\n31,"""EDWARD WINSLOW, WILLIAM BREWSTER, ISAAC ALLERTON (LEIDEN LEADERS)""","""leaders in the group merging ships""","""PILGRIMS""",9.0,2\r\n32,"""WILLIAM BRADFORD, EDWARD WINSLOW, WILLIAM BREWSTER, ISAAC ALLERTON (LEIDEN LEADERS)""","""part of the group after merging ships""","""LEIDEN CONGREGATION""",9.0,2\r\n33,"""CARVER (WEALTHY INVESTOR)""","""invested personal fortune in the voyage""","""MAYFLOWER VOYAGE INVESTMENT""",9.0,2\r\n34,"""GILES HEALE (SURGEON)""","""witness to death-bed will""","""WILLIAM MULLINS (PASSENGER)""",9.0,2\r\n35,"""CARVER (VESSEL ORGANIZER)""","""organized voyage and negotiations with funders""","""WESTON AND MERCHANT ADVENTURERS""",8.0,2\r\n36,"""WILLIAM MULLINS (FAMILY)""","""perished in first winter with family""","""ALL PASSENGERS OF MAYFLOWER""",8.0,2\r\n37,"""FORECASTLE SPACE""","""space for various roles""","""SHIP\'S COOK, SHIP\'S SURGEON, AND SHIP\'S OFFICERS""",8.0,2\r\n38,"""MASTER CHRISTOPHER JONES (CABIN)""","""measuring""","""TEN BY SEVEN FEET""",7.0,2\r\n39,"""CAPSTAN""","""used to pull in ropes or cables""","""VERTICAL AXLE""",4.0,2\r\n\n```\n-----Sources-----\n```csv\nid,content\r\n0,"to the Separatist Church in Nottinghamshire England who came to Leiden, Holland about 1608 and became prominent in the church there. He came on the ""Mayflower"" with his wife Dorothy, leaving a young son in Leiden; Dorothy drowned while the ship was at anchor in Cape Cod Harbor. He became colony Governor after the death of John Carver, and was prominent in the Plymouth Church. His writings of early Plymouth Colony are important historic documents. Edward Winslow - A gentleman from a well-off family who was prominent in the Separatist church in Leiden and involved with Brewster in printing anti-Anglican\nPassage 10: George Soule (Mayflower passenger) George Soule (c. 1601 – between 20 September 1677 and 22 January 1679) was a colonist who was one of the indentured servants on the ""Mayflower"" and helped establish Plymouth Colony in 1620. He was one of the signers of the Mayflower Compact. It is known that George came on the ""Mayflower"" and was credited to the household of Edward Winslow as a manservant or apprentice, along with Elias Story and a little girl Ellen More, who both died in the first winter. What is not generally understood is the long-term effect of William Brewster\'s scheme"\r\n1,"Passage 1: Aft on the main deck in the stern was the cabin for Master Christopher Jones, measuring about ten by seven feet (3 m × 2.1 m).  Forward of that was the steerage room, which housed a whipstaff (tiller extension) for sailing control; not a wheel, as in later ships. Also here was the ship\'s compass and probably also berths for the ship\'s officers. Forward of the steerage room was the capstan, a vertical axle used to pull in ropes or cables. Far forward on the main deck, just aft of the bow, was the forecastle space where the ship\'s cook\nPassage 2: the ship\'s surgeon, a young man just out of apprenticeship as a London Barber-Surgeon by the name of Giles Heale. His name appears as a witness to the death-bed will of William Mullins in February 1621. Another person that Bradford also did not mention who is recorded as possibly being a principal officer of the Mayflower due to his title, is a man identified only as ""Master"" Leaver. He is recorded in Mourt\'s Relation (1622) as rescuing Pilgrims lost in a forest in January 1621. ""Mayflower"" embarked about sixty-five passengers in London about the middle of July 1620, proceeded to\nPassage 3: Carver and his wife Katherine boarded ""Mayflower"" with five servants and seven year-old Jasper More, one of the four children of the More family who were sent in the care of the Pilgrims. Carver seems to have been elected governor of the ""Mayflower"" for the duration of the Atlantic crossing. The ""Mayflower"" anchored off Cape Cod in November, 1620, and the Mayflower Compact was signed aboard ship on November 11; it became the first governing document for Plymouth Colony. Carver may have been the author of the Compact, and was definitely its first signer. He was subsequently chosen to be\nPassage 4: John Alden Capt. John Alden Sr. ( 15981687) was a crew member on the historic 1620 voyage of the Pilgrim ship ""Mayflower"". Rather than return to England with the ship, he stayed at what became Plymouth Colony. He was hired in Southampton, England, as the ship\'s cooper, responsible for maintaining the ship\'s barrels. He was a signatory to the Mayflower Compact. He married fellow ""Mayflower"" passenger Priscilla Mullins, whose entire family perished in the first winter. He served in a number of important government positions such as Assistant Governor, Duxbury Deputy to the General Court of Plymouth, Captain Myles Standish\'s\nPassage 5: supremacy of the king and the Church of England. To fund the ""Mayflower"" voyage, the Leiden congregation turned to Thomas Weston and the Merchant Adventurers, London businessmen interested in supporting the voyage in hopes of profit. Carver had the task of organizing the voyage and negotiating funding with Weston and the Adventurers, along with Cushman as the chief agent. Carver was in Southampton in June 1620 purchasing supplies for the ""Mayflower"" voyage, along with Christopher Martin. Carver was very wealthy and provided much of his personal fortune to make investment in the joint-stock company and in the Mayflower voyage itself.\nPassage 6: a week, in an effect to increase profits, without such as due time for religious activities. The Pilgrims balked at this and refused to agree to the new terms. William Mullins played a part in these deliberations, probably because he had a large investment and needed to ensure a satisfactory return on it, as an Adventurers member. And although Robert Cushman, who had been the Leiden agent for ""Mayflower"" voyage preparations, came to Plymouth in November 1621 to try to settle the rift between the Pilgrims and the Adventurers, it was never resolved. Eventually the Pilgrims bought out the Adventurers\nPassage 7: had been designated by the Merchant Adventurers to act as shipboard governor during the trans-Atlantic trip; and Stephen Hopkins, a veteran of a failed colonial venture that may have inspired Shakespeare\'s ""The Tempest"". The group who later became the Leiden Leaders after the merging of ships included John Carver, William Bradford, Edward Winslow, William Brewster, and Isaac Allerton. The ""Mayflower"" departed Plymouth, England on September 6, 1620 with 102 passengers and about 30 crew members in the small, 106 foot-long ship. The seas were not severe during the first month in the Atlantic but, by the second month, the ship\nPassage 8: Plymouth Colony. He left his family in Leiden and came on the ""Mayflower"" with only young servant William Butten, who died at sea a few days before reaching Cape Cod. He was the largely self-taught physician and surgeon of the colony and died in 1633 of an infectious fever that killed many that year. Christopher Martin - He was a prosperous leader of those non-religious persons known as ""Strangers"" on the ""Mayflower"", as well as a representative of the Merchant Adventurer investment group. He came on the ship with his wife and two servants, one of whom was his step-son\nPassage 9: to the Separatist Church in Nottinghamshire England who came to Leiden, Holland about 1608 and became prominent in the church there. He came on the ""Mayflower"" with his wife Dorothy, leaving a young son in Leiden; Dorothy drowned while the ship was at anchor in Cape Cod Harbor. He became colony Governor after the death of John Carver, and was prominent in the Plymouth Church. His writings of early Plymouth Colony are important historic documents. Edward Winslow - A gentleman from a"\r\n\n```\n'

Output:
'\n-----Relationships-----\n```csv\nid,source entity,relationship between the source entity and the target entity,target entity,weight,rank\r\n14,"""CHRISTOPHER MARTIN (STRANGER AND MERCHANT ADVENTURER REPRESENTATIVE)""","""came on the ship with family and servants""","""MAYFLOWER""",3.0,7\r\n15,"""CARVER AND HIS WIFE KATHERINE (FAMILY)""","""boarded with servants, child, and served as governor during crossing""","""MAYFLOWER""",3.0,6\r\n17,"""MAYFLOWER""","""departed on this date with passengers and crew""","""SEPTEMBER 6, 1620 DEPARTURE DATE""",5.0,6\r\n18,"""MAYFLOWER""","""embarked about""","""65 PASSENGERS""",5.0,6\r\n19,"""MASTER LEAVER (IDENTIFIED AS POSSIBLE PRINCIPAL OFFICER)""","""possibly a principal officer of the ship""","""MAYFLOWER""",1.0,6\r\n24,"""JOHN ALDEN SR.""","""signed the compact as a crew member""","""MAYFLOWER COMPACT (SIGNATORY)""",3.0,5\r\n26,"""JOHN CARVER (DECEASED GOVERNOR)""","""initial elected governor for Atlantic crossing""","""MAYFLOWER COMPACT SIGNATORY""",3.0,3\r\n30,"""THOMAS WESTON AND MERCHANT ADVENTURERS, LONDON BUSINESSMEN""","""provided funding for the voyage""","""MAYFLOWER VOYAGE FUNDING""",2.0,2\r\n31,"""EDWARD WINSLOW, WILLIAM BREWSTER, ISAAC ALLERTON (LEIDEN LEADERS)""","""leaders in the group merging ships""","""PILGRIMS""",2.0,2\r\n35,"""CARVER (VESSEL ORGANIZER)""","""organized voyage and negotiations with funders""","""WESTON AND MERCHANT ADVENTURERS""",3.0,2\r\n38,"""MASTER CHRISTOPHER JONES (CABIN)""","""measuring""","""TEN BY SEVEN FEET""",10.0,2\r\n5,"""JOHN CARVER (DECEASED GOVERNOR)""","""became the first Governor after voyage""","""PLYMOUTH COLONY""",2.0,10\r\n6,"""WILLIAM BRADFORD (GOVERNOR)""","""prominent leader and founder of the colony""","""PLYMOUTH COLONY""",2.0,9\r\n7,"""STEPHEN HOPKINS (VETERAN OF FAILED COLONIAL VENTURE)""","""part of the group merging ships with Leiden Leaders""","""PLYMOUTH COLONY""",2.0,9\r\n12,"""EDWARD WINSLOW""","""took over as Governor after death""","""JOHN CARVER""",2.0,7\r\n\n```\n-----Sources-----\n```csv\nid,content\r\n0,"Master Christopher Jones was the commander of the Mayflower, with a private cabin located at the stern of the main deck, measuring approximately 10 by 7 feet. Adjacent to his cabin was the steerage room, housing essential navigation equipment like the ship’s compass and a whipstaff for steering. The area also included the capstan, used for hauling ropes, and forward sections like the forecastle, where the cook worked. His role as the ship’s master placed him in charge of navigation and overall operations during the 1620 voyage to Plymouth Colony. No further personal details or direct involvement in colonial governance are noted in the provided text."\r\n\n```\n'<|end|>

---Example 2---

Question:
Where was the director of film French Heels born?

Data tables:
'\n-----Relationships-----\n```csv\nid,source entity,relationship between the source entity and the target entity,target entity,weight,rank\r\n0,"""BRIAN PATRICK KENNEDY""","""is currently the director of""","""PEABODY ESSEX MUSEUM""",10.0,5\r\n1,"""FRENCH HEELS""","""directed by""","""EDWIN L. HOLLYWOOD""",8.0,5\r\n2,"""OLAV AARAAS""","""has been the director since 2001""","""NORWEGIAN MUSEUM OF CULTURAL HISTORY""",8.0,5\r\n3,"""OLAV AARAAS""","""is a ""","""NORWEGIAN HISTORIAN AND MUSEUM DIRECTOR""",8.0,5\r\n4,"""BRIAN PATRICK KENNEDY""","""was the director from 2010 to 2019""","""TOLEDO MUSEUM OF ART""",8.0,5\r\n5,"""OLAV AARAAS""","""was the director from 1993 to 2010""","""MAIHAUGEN""",7.0,5\r\n6,"""BRIAN PATRICK KENNEDY""","""was the director from 2005 to 2010""","""HOOD MUSEUM OF ART""",7.0,5\r\n7,"""OLAV AARAAS""","""was the director from 1982 to 1993""","""SOGN FOLK MUSEUM""",6.0,5\r\n8,"""BRIAN PATRICK KENNEDY""","""was the director in Canberra from 1997-2004""","""NATIONAL GALLERY OF AUSTRALIA""",6.0,5\r\n9,"""EDWIN L. HOLLYWOOD""","""directed by him ""","""1922 AMERICAN SILENT ROMANTIC COMEDY FILM \'FRENCH HEELS\'""",8.0,4\r\n10,"""EDWIN L. HOLLYWOOD""","""in charge of starring Harry Morey""","""VITAGRAPH\'S FILM UNIT""",8.0,4\r\n11,"""S.N. MATHUR""","""was the Director from September 1975 to February 1980""","""INDIAN INTELLIGENCE BUREAU""",6.0,4\r\n12,"""S.N. MATHUR""","""was between certain years""","""DIRECTOR OF THE INDIAN INTELLIGENCE BUREAU""",6.0,4\r\n13,"""S.N. MATHUR""","""was Director General of in Punjab""<SEP>""was the Director General of between certain years""","""PUNJAB POLICE FORCE""",5.0,4\r\n14,"""DANA BLANKSTEIN- COHEN""","""is the director of""","""ISRAELI ACADEMY OF FILM AND TELEVISION""",10.0,3\r\n15,"""DANA BLANKSTEIN- COHEN""","""is the director of ""","""ISRAELI ACADEMY OF FILM AND TELEVISION DIRECTOR""",8.0,3\r\n16,"""JESSE EDWARD HOBSON""","""was the director from 1947 to 1955""","""SRI INTERNATIONAL""",8.0,3\r\n17,"""PETER LEVIN""","""works in as a director""","""FILM, TELEVISION AND THEATRE INDUSTRY""",7.0,3\r\n18,"""JASON MOORE""","""works in as a director ""","""FILM, THEATRE AND TELEVISION INDUSTRY DIRECTOR""",7.0,3\r\n19,"""JASON MOORE""","""works in as a director""","""FILM, THEATRE AND TELEVISION INDUSTRY""",7.0,3\r\n20,"""PETER LEVIN""","""works in as a director""","""AMERICAN FILM, TELEVISION AND THEATRE INDUSTRY""",7.0,3\r\n21,"""IAN BARRY""","""works in as a director""","""FILM AND TV INDUSTRY""",7.0,3\r\n22,"""IAN BARRY""","""works in as a director ""","""AUSTRALIAN FILM AND TV INDUSTRY""",7.0,3\r\n23,"""FRENCH HEELS""","""starred in""","""IRENE CASTLE""",6.0,3\r\n24,"""JESSE EDWARD HOBSON""","""was the director prior to SRI International ""<SEP>""was the director prior to SRI International""","""ARMOUR RESEARCH FOUNDATION""",6.0,3\r\n\n```\n-----Sources-----\n```csv\nid,content\r\n0,"Passage 1: Brian Patrick Kennedy( born 5 November 1961) is an Irish- born art museum director who has worked in Ireland and Australia, and now lives and works in the United States.\nHe is currently the director of the Peabody Essex Museum.\nHe was the director of the Toledo Museum of Art in Ohio from 2010 to 2019.\nHe was the director of the Hood Museum of Art from 2005 to 2010, and the National Gallery of Australia( Canberra) from 1997- 2004.\nPassage 2: Olav Aaraas( born 10 July 1950) is a Norwegian historian and museum director.\nHe was born in Fredrikstad.\nFrom 1982 to 1993 he was the director of Sogn Folk Museum, from 1993 to 2010 he was the director of Maihaugen and from 2001 he has been the director of the Norwegian Museum of Cultural History.\nIn 2010 he was decorated with the Royal Norwegian Order of St. Olav.\nPassage 3: S.N. Mathur was the Director of the Indian Intelligence Bureau between September 1975 and February 1980.\nHe was also the Director General of Police in Punjab.\nPassage 4: Peter Levin is an American director of film, television and theatre.\nPassage 5: French Heels is a lost 1922 American silent romantic comedy film directed by Edwin L. Hollywood and starring Irene Castle.\nBased on short story"" Knots and Windshakes"" by Clarence Budington Kelland which appeared in"" Everybody\'s Magazine"", it was distributed by W. W. Hodkinson.\nPassage 6: Dana Blankstein- Cohen( born March 3, 1981) is the director of the Israeli Academy of Film and Television.\nShe is a film director, and an Israeli culture entrepreneur.\nPassage 7: Ian Barry is an Australian director of film and TV.\nPassage 8: Edwin L. Hollywood (October 9, 1892 – May 15, 1958) was an American actor and film director.\nHe was born in New York City.\nHollywood was in charge of Vitagraph\'s film unit that starred Harry Morey.\nHe died in Glendale, California.\nPassage 9: Jason Moore( born October 22, 1970) is an American director of film, theatre and television.\nPassage 10: Jesse Edward Hobson( May 2, 1911 – November 5, 1970) was the director of SRI International from 1947 to 1955.\nPrior to SRI, he was the director of the Armour Research Foundation."\r\n\n```\n'

Output:
'\n-----Relationships-----\n```csv\nid,source entity,relationship between the source entity and the target entity,target entity,weight,rank\r\n1,"""FRENCH HEELS""","""directed by""","""EDWIN L. HOLLYWOOD""",10.0,5\r\n9,"""EDWIN L. HOLLYWOOD""","""directed by him ""","""1922 AMERICAN SILENT ROMANTIC COMEDY FILM \'FRENCH HEELS\'""",9.0,4\r\n10,"""EDWIN L. HOLLYWOOD""","""in charge of starring Harry Morey""","""VITAGRAPH\'S FILM UNIT""",7.0,4\r\n8,"""EDWIN L. HOLLYWOOD""","""born in""","""NEW YORK CITY""",10.0,5\r\n23,"""FRENCH HEELS""","""starred in""","""IRENE CASTLE""",5.0,3\r\n0,"""BRIAN PATRICK KENNEDY""","""is currently the director of""","""PEABODY ESSEX MUSEUM""",2.0,5\r\n2,"""OLAV AARAAS""","""has been the director since 2001""","""NORWEGIAN MUSEUM OF CULTURAL HISTORY""",2.0,5\r\n4,"""BRIAN PATRICK KENNEDY""","""was the director from 2010 to 2019""","""TOLEDO MUSEUM OF ART""",2.0,5\r\n5,"""OLAV AARAAS""","""was the director from 1993 to 2010""","""MAIHAUGEN""",2.0,5\r\n6,"""BRIAN PATRICK KENNEDY""","""was the director from 2005 to 2010""","""HOOD MUSEUM OF ART""",2.0,5\r\n14,"""DANA BLANKSTEIN- COHEN""","""is the director of""","""ISRAELI ACADEMY OF FILM AND TELEVISION""",2.0,3\r\n16,"""JESSE EDWARD HOBSON""","""was the director from 1947 to 1955""","""SRI INTERNATIONAL""",2.0,3\r\n17,"""PETER LEVIN""","""works in as a director""","""FILM, TELEVISION AND THEATRE INDUSTRY""",2.0,3\r\n21,"""IAN BARRY""","""works in as a director""","""FILM AND TV INDUSTRY""",2.0,3\r\n\n```\n-----Sources-----\n```csv\nid,content\r\n0,"Edwin L. Hollywood, director of the 1922 silent film *French Heels*, was born in New York City. He began his career as an actor before transitioning to directing, notably leading Vitagraph\'s film unit featuring Harry Morey. The romantic comedy starred Irene Castle and was based on a Clarence Budington Kelland short story. No other biographical details about Hollywood’s early life or education are provided in the text."\r\n\n```\n'<|end|>

---Your turn---
Only give me the output. Never explain, justify, or add commentary!!!

Question:
{input_question}

Data tables:
'{input_data_table}'

Output:
'''

PROMPTS["question_planning_2.0"]='''
You possess a strong ability to break down problems. Your task is to, for the given original question, step by step derive its thinking process and the sub-questions extracted from the original question, and write down a simple process for obtaining the final answer. Here, "Get_Answer" represents the pseudo-function to be called, and "Final_Answer" indicates the final answer. Please note that if the original question does not require any decomposition, please refer to the following "Example 0". And if the original question needs to be decomposed into more than three steps, then it is directly regarded as a question that does not require decomposition. Below are some examples:

###################
# Example 0:
###################

Original_Question: str = "What is the ethnic group of Booker T. Jones?"

Output:
Thought1: str = "An atomic question, no need to decompose. Search directly."
Sub_Question_1: str = "What is the ethnic group of Booker T. Jones?"
Ans_1: str = Get_Answer(Sub_Question_1)
Final_Answer: str = Ans_1

###################
# Example 1:
###################

Original_Question: str = "Who succeeded the first President of Namibia?"

Output:
Thought1: str = "If I want to know who succeeded the first President of Namibia, I need to first know who is the first President of Namibia."
Sub_Question_1: str = "Who is the first President of Namibia?"
Ans_1: str = Get_Answer(Sub_Question_1)
Thought2: str = "After knowing who is the first President of Namibia, I need to know who succeeded him."
Sub_Question_2: str = f"Who succeeded {Ans_1}?"
Ans_2: str = Get_Answer(Sub_Question_2)
Final_Answer: str = Ans_2

###################
# Example 2:
###################

Original_Question: str = "What is the foundational text of Android developer's country?"

Output:
Thought1: str = "If I want to know what is the foundational text of Android developer's country, I need to first know what(who) is the developer of Android."
Sub_Question_1: str = "Who is the developer of Android?"
Ans_1: str = Get_Answer(Sub_Question_1)
Thought2: str = "After knowing who is the developer of Android, I need to the know its country."
Sub_Question_2: str = f"What is the country of {Ans_1}?"
Ans_2: str = Get_Answer(Sub_Question_2)
Thought3: str = "After knowing what is the country of Android developer, I need to know what is the foundational text of country Ans_2."
Sub_Question_3: str = f"What is the foundational text of {Ans_2}?"
Ans_3: str = Get_Answer(Sub_Question_3)
Final_Answer: str = Ans_3

###################
# Example 3:
###################

Original_Question: str = "When was the first establishment that McDonaldization is named after, open in the country Horndean is located?"

Output:
Thought1: str = "If I want to know when the first establishment that McDonaldization is named after, open in the country Horndean is located, I need to first know Where is the country Horndean located."
Sub_Question_1: str = "Where is the country Horndean located?"
Ans_1: str = Get_Answer(Sub_Question_1)
Thought2: str = "After knowing where is the country Horndean located, I need to the know what is the first establishment that McDonaldization is named after in Ans_1."
Sub_Question_2: str = f"what is the first establishment that McDonaldization is named after in {Ans_1}?"
Ans_2: str = Get_Answer(Sub_Question_2)
Thought3: str = "After knowing what is the first establishment that McDonaldization is named after in Ans_1, I need to know when did Ans_2 open in Ans_1."
Sub_Question_3: str = f"When did {Ans_2} open in {Ans_1}?"
Ans_3: str = Get_Answer(Sub_Question_3)
Final_Answer: str = Ans_3

###################
# Example 4:
###################

Original_Question: str = "Which magazine was started first Arthur's Magazine or First for Women?"

Output:
Thought1: str = "If I want to know which magazine was started first, I need to first know when Arthur's Magazine was started."
Sub_Question_1: str = "When was Arthur's Magazine started?"
Ans_1: str = Get_Answer(Sub_Question_1)
Thought2: str = "At the same time, I need to know when First for Women was started."
Sub_Question_2: str = "When was First for Women started?"
Ans_2: str = Get_Answer(Sub_Question_2)
Thought3: str = "After knowing when Arthur's Magazine was started and when First for Women was started, I need to know which magazine was started first given that Arthur's Magazine was started in Ans_1 and First for Women was started in Ans_2."
Sub_Question_3: str = f"Which magazine was started first given that Arthur's Magazine was started in {Ans_1} and First for Women was started in {Ans_2}?"
Ans_3: str = Get_Answer(Sub_Question_3)
Final_Answer: str = Ans_3

###################
# Example 5:
###################

Original_Question: str = "Which areas border with Burlington County and Trumbull County at the same time?"

Output:
Thought1: str = "If I want to know which areas border with Burlington County and Trumbull County at the same time, I need to first know which areas border with Burlington County."
Sub_Question_1: str = "Which areas border with Burlington County?"
Ans_1: str = Get_Answer(Sub_Question_1)
Thought2: str = "At the same time, I need to know which areas border with Trumbull County."
Sub_Question_2: str = "Which areas border with Trumbull County?"
Ans_2: str = Get_Answer(Sub_Question_2)
Thought3: str = "After knowing which areas border with Burlington County and which areas border with Trumbull County , I need to know which areas border with Burlington County and Trumbull County at the same time given that Ans_1 border with Burlington County and Ans_2 border with Trumbull County."
Sub_Question_3: str = f"Which areas border with Burlington County and Trumbull County at the same time given that {Ans_1} border with Burlington County and {Ans_2} border with Trumbull County?"
Ans_3: str = Get_Answer(Sub_Question_3)
Final_Answer: str = Ans_3

###################
# Example 6:
###################

Original_Question: str = "What are the same genre shared between Alice in Wonderland, Blues Brothers 2000 and Pinocchio?"

Output:
Thought1: str = "The current question requires more than three steps to break down, so output the original question directly."
Sub_Question_1: str = "What are the same genre shared between Alice in Wonderland, Blues Brothers 2000 and Pinocchio?"
Ans_1: str = Get_Answer(Sub_Question_1)
Final_Answer: str = Ans_1

###################
# Example 7:
###################

Original_Question: str = "What are the political party of people who are cast members of both The Blues Brothers and Going My Way?"

Output:
Thought1: str = "If I want to know what are the political party of people who are cast members of both The Blues Brothers and Going My Way, I need to first know who are cast members of both The Blues Brothers and Going My Way."
Sub_Question_1: str = "Who are cast members of both The Blues Brothers and Going My Way?"
Ans_1: str = Get_Answer(Sub_Question_1)
Thought2: str = "After knowing who are cast members of both The Blues Brothers and Going My Way, I need to know what are the political party of Ans_1."
Sub_Question_2: str = "What are the political party of {Ans_1}?"
Ans_2: str = Get_Answer(Sub_Question_2)
Final_Answer: str = Ans_2

###################
# Example 8:
###################

Original_Question: str = "Prior to playing for Michigan State, Keith Nichol played football for a school located in what city?"

Output:
Thought1: str = "If I want to know prior to playing for Michigan State, Keith Nichol played football for a school located in what city, I first need to know which school Keith Nichol played for before joining Michigan State."
Sub_Question_1: str = "Which school did Keith Nichol play football for prior to playing for Michigan State?"
Ans_1: str = Get_Answer(Sub_Question_1)
Thought2: str = "After knowing which school did Keith Nichol play football for prior to playing for Michigan State, I need to know In what city is Ans_1 located."
Sub_Question_2: str = f"In what city is {Ans_1} located?"
Ans_2: str = Get_Answer(Sub_Question_2)
Final_Answer: str = Ans_2

###################
# Your turn! Imitate the above example to output only the following thought process, do not return anything else.
###################

{_original_question_}

Output:
'''



PROMPTS["summary_response_2.0"]='''
You are an expert in problem thinking and answering. You will be given a context (The data tables that contain relationships and sources) and a question. Your task is to first output a chain of thought based on the context, detailing how you think and analyze the problem step by step according to the context, and then give the final answer. When presenting the chain of thought, adopt the thinking mode of "Because..., therefore...", and express your logical reasoning process clearly.
Please note the following points:
1. Elaborate on your thinking process when understanding the problem and analyzing the context.
2. If there are contradictions or inconsistencies related to the question in the context, clearly point them out and explain how you handle them.
3. Describe how you extract key information from the context and use it to construct the answer.
4. If you need to make inferences or assumptions, clearly explain your reasoning process and the basis for your assumptions.
5. The final answer should be concise and clear, directly answering the question, and must be based on the analysis in the context and the chain of thought.
6. Please add '<|start|>' before and '<|end|>' after the final answer.

#####Here is an example: #####

Question:
Where was the director of film French Heels born?

Data tables:
'\n-----Relationships-----\n```csv\nid,source entity,relationship between the source entity and the target entity,target entity,weight,rank\r\n0,"""BRIAN PATRICK KENNEDY""","""is currently the director of""","""PEABODY ESSEX MUSEUM""",10.0,5\r\n1,"""FRENCH HEELS""","""directed by""","""EDWIN L. HOLLYWOOD""",8.0,5\r\n2,"""OLAV AARAAS""","""has been the director since 2001""","""NORWEGIAN MUSEUM OF CULTURAL HISTORY""",8.0,5\r\n3,"""OLAV AARAAS""","""is a ""","""NORWEGIAN HISTORIAN AND MUSEUM DIRECTOR""",8.0,5\r\n4,"""BRIAN PATRICK KENNEDY""","""was the director from 2010 to 2019""","""TOLEDO MUSEUM OF ART""",8.0,5\r\n5,"""OLAV AARAAS""","""was the director from 1993 to 2010""","""MAIHAUGEN""",7.0,5\r\n6,"""BRIAN PATRICK KENNEDY""","""was the director from 2005 to 2010""","""HOOD MUSEUM OF ART""",7.0,5\r\n7,"""OLAV AARAAS""","""was the director from 1982 to 1993""","""SOGN FOLK MUSEUM""",6.0,5\r\n8,"""BRIAN PATRICK KENNEDY""","""was the director in Canberra from 1997-2004""","""NATIONAL GALLERY OF AUSTRALIA""",6.0,5\r\n9,"""EDWIN L. HOLLYWOOD""","""directed by him ""","""1922 AMERICAN SILENT ROMANTIC COMEDY FILM \'FRENCH HEELS\'""",8.0,4\r\n10,"""EDWIN L. HOLLYWOOD""","""in charge of starring Harry Morey""","""VITAGRAPH\'S FILM UNIT""",8.0,4\r\n11,"""S.N. MATHUR""","""was the Director from September 1975 to February 1980""","""INDIAN INTELLIGENCE BUREAU""",6.0,4\r\n12,"""S.N. MATHUR""","""was between certain years""","""DIRECTOR OF THE INDIAN INTELLIGENCE BUREAU""",6.0,4\r\n13,"""S.N. MATHUR""","""was Director General of in Punjab""<SEP>""was the Director General of between certain years""","""PUNJAB POLICE FORCE""",5.0,4\r\n14,"""DANA BLANKSTEIN- COHEN""","""is the director of""","""ISRAELI ACADEMY OF FILM AND TELEVISION""",10.0,3\r\n15,"""DANA BLANKSTEIN- COHEN""","""is the director of ""","""ISRAELI ACADEMY OF FILM AND TELEVISION DIRECTOR""",8.0,3\r\n16,"""JESSE EDWARD HOBSON""","""was the director from 1947 to 1955""","""SRI INTERNATIONAL""",8.0,3\r\n17,"""PETER LEVIN""","""works in as a director""","""FILM, TELEVISION AND THEATRE INDUSTRY""",7.0,3\r\n18,"""JASON MOORE""","""works in as a director ""","""FILM, THEATRE AND TELEVISION INDUSTRY DIRECTOR""",7.0,3\r\n19,"""JASON MOORE""","""works in as a director""","""FILM, THEATRE AND TELEVISION INDUSTRY""",7.0,3\r\n20,"""PETER LEVIN""","""works in as a director""","""AMERICAN FILM, TELEVISION AND THEATRE INDUSTRY""",7.0,3\r\n21,"""IAN BARRY""","""works in as a director""","""FILM AND TV INDUSTRY""",7.0,3\r\n22,"""IAN BARRY""","""works in as a director ""","""AUSTRALIAN FILM AND TV INDUSTRY""",7.0,3\r\n23,"""FRENCH HEELS""","""starred in""","""IRENE CASTLE""",6.0,3\r\n24,"""JESSE EDWARD HOBSON""","""was the director prior to SRI International ""<SEP>""was the director prior to SRI International""","""ARMOUR RESEARCH FOUNDATION""",6.0,3\r\n\n```\n-----Sources-----\n```csv\nid,content\r\n0,"Passage 1: Brian Patrick Kennedy( born 5 November 1961) is an Irish- born art museum director who has worked in Ireland and Australia, and now lives and works in the United States.\nHe is currently the director of the Peabody Essex Museum.\nHe was the director of the Toledo Museum of Art in Ohio from 2010 to 2019.\nHe was the director of the Hood Museum of Art from 2005 to 2010, and the National Gallery of Australia( Canberra) from 1997- 2004.\nPassage 2: Olav Aaraas( born 10 July 1950) is a Norwegian historian and museum director.\nHe was born in Fredrikstad.\nFrom 1982 to 1993 he was the director of Sogn Folk Museum, from 1993 to 2010 he was the director of Maihaugen and from 2001 he has been the director of the Norwegian Museum of Cultural History.\nIn 2010 he was decorated with the Royal Norwegian Order of St. Olav.\nPassage 3: S.N. Mathur was the Director of the Indian Intelligence Bureau between September 1975 and February 1980.\nHe was also the Director General of Police in Punjab.\nPassage 4: Peter Levin is an American director of film, television and theatre.\nPassage 5: French Heels is a lost 1922 American silent romantic comedy film directed by Edwin L. Hollywood and starring Irene Castle.\nBased on short story"" Knots and Windshakes"" by Clarence Budington Kelland which appeared in"" Everybody\'s Magazine"", it was distributed by W. W. Hodkinson.\nPassage 6: Dana Blankstein- Cohen( born March 3, 1981) is the director of the Israeli Academy of Film and Television.\nShe is a film director, and an Israeli culture entrepreneur.\nPassage 7: Ian Barry is an Australian director of film and TV.\nPassage 8: Edwin L. Hollywood (October 9, 1892 – May 15, 1958) was an American actor and film director.\nHe was born in New York City.\nHollywood was in charge of Vitagraph\'s film unit that starred Harry Morey.\nHe died in Glendale, California.\nPassage 9: Jason Moore( born October 22, 1970) is an American director of film, theatre and television.\nPassage 10: Jesse Edward Hobson( May 2, 1911 – November 5, 1970) was the director of SRI International from 1947 to 1955.\nPrior to SRI, he was the director of the Armour Research Foundation."\r\n\n```\n'

Output:
Because the question asks for the birthplace of the director of the film French Heels, we first identify the director.
Because Relationship ID 1 states that French Heels was "directed by" Edwin L. Hollywood, confirming his role as director.
Because Source Passage 5 describes French Heels as directed by Edwin L. Hollywood, and Source Passage 8 explicitly states: "Edwin L. Hollywood [...] was born in New York City."
Because no other entries in the Relationships or Sources contradict this information or associate another director with French Heels.
Therefore, the birthplace of Edwin L. Hollywood, the director of French Heels, is New York City.
Final answer: <|start|>New York City<|end|>

#####Your turn: #####

Question:
{input_question}

Data tables:
'{input_data_table}'

Output:
'''

PROMPTS["summary_response_3.0"]='''
You are an expert in problem thinking and answering. You will be given a context (The data tables that contain relationships and sources) and a question. Your task is to first output a chain of thought based on the context, detailing how you think and analyze the question step by step according to the context, and then give the final answer. When presenting the chain of thought, adopt the thinking mode of "Because..., therefore...", and express your logical reasoning process clearly.
Please note the following points:
1. Elaborate on your thinking process when understanding the problem and analyzing the context.
2. If there are contradictions or inconsistencies related to the question in the context, clearly point them out and explain how you handle them.
3. Describe how you extract key information from the context and use it to construct the answer.
4. If you need to make inferences or assumptions, clearly explain your reasoning process and the basis for your assumptions.
5. The final answer should be concise , clear and no more than one sentence. And the final answer must be based on the analysis in the context and your chain of thought.
6. Please add '<|start|>' before and '<|end|>' after the final answer.

Use the following form:

Output:
<reasoning>
Because ...
Therefore ...
...
</reasoning>
Final answer: <|start|>...<|end|>

#####Your turn: #####

Question:
{input_question}

Data tables:
'{input_data_table}'

Output:
'''

PROMPTS["summary_response_RL"]='''
You are an expert in problem thinking and answering. You will be given a context (The data tables that contain relationships and sources) and a question. Your task is to first output a chain of thought based on the context, detailing how you think and analyze the problem step by step according to the context, and then give the final answer. When presenting the chain of thought, adopt the thinking mode of "Because..., therefore...", and express your logical reasoning process clearly.
Please note the following points:
1. Elaborate on your thinking process when understanding the problem and analyzing the context.
2. If there are contradictions or inconsistencies related to the question in the context, clearly point them out and explain how you handle them.
3. Describe how you extract key information from the context and use it to construct the answer.
4. If you need to make inferences or assumptions, clearly explain your reasoning process and the basis for your assumptions.
5. The final answer should be concise and clear, directly answering the question, and must be based on the analysis in the context and the chain of thought.
6. Please add '<|start|>' before and '<|end|>' after the final answer.

#####Your turn: #####

Question:
{input_question}

Data tables:
'{input_data_table}'

Output:
'''