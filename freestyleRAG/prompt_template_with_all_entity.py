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
- relationship_description: use a short statement to explain the relationship between the source and target entities.
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter})

2. Treat each entity identified in step 1 as a special relationship, then the source entity name of this entity is the same as the target entity, and the relationship description becomes a simple description of this entity, find them and extract the following information:
- source_entity: name of this entity
- target_entity: name of this entity
- relationship_description: use a short statement to describe this entity.
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter})

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in the same language as the text document as a single list of all the relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

Here are some examples:

Example 1:

Text: while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths

Output:
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"frustration"{tuple_delimiter}"experiences"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"interacts with"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"authoritarian certainty"{tuple_delimiter}"represents"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"opposes vision of control and order"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Cruz"{tuple_delimiter}"control and order"{tuple_delimiter}"represents"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The device"{tuple_delimiter}"shows reverence towards"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"The device"{tuple_delimiter}"technology"{tuple_delimiter}"is associated with"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"technology"{tuple_delimiter}"game"{tuple_delimiter}"could change"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"All people"{tuple_delimiter}"All people"{tuple_delimiter}"were brought here by different paths"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Alex"{tuple_delimiter}"main character and observer"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Taylor"{tuple_delimiter}"authoritative figure with layers of complexity"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Jordan"{tuple_delimiter}"committed to discovery and unity"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Cruz"{tuple_delimiter}"Cruz"{tuple_delimiter}"symbolizes control and narrow vision"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"The device"{tuple_delimiter}"The device"{tuple_delimiter}"symbol of opportunity and transformation"{tuple_delimiter}){record_delimiter}
("content_keywords"{tuple_delimiter}"power dynamics, discovery, control, transformation, competition, unspoken alliances"{tuple_delimiter}){completion_delimiter}

Example 2:

Text: I acknowledge with deep gratitude the dozens of friends, colleagues, and perfect strangers who helped me explore the benefits of indigenous diets. Without their contributions, this project could never have happened.\n\nFirst and foremost, I want to thank my beloved co-adventurer (and husband) Ross Levy. His patience and support allowed this book to move beyond a mere daydream.

I would also like to thank my children, Arlen and Emet Levy, who bravely tasted all the recipes and who never failed to give their honest opinions. I heartily thank Allison Fragakis for her excellent nutrition advice throughout this project and for spearheading the recipe-testing portion of this book.\n\nI am deeply grateful to my parents, Susan and David Miller, whose own wanderlust and love of eating and cooking first launched me into the world of travel adventure and indigenous foods and to my brother Sam Miller, the armchair nutritionist, who seems to know more than many professionals.

Output:
("relationship"{tuple_delimiter}"Author"{tuple_delimiter}"indigenous diets"{tuple_delimiter}"explores benefits of"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Author"{tuple_delimiter}"friends"{tuple_delimiter}"expresses gratitude to"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Author"{tuple_delimiter}"colleagues"{tuple_delimiter}"expresses gratitude to"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Author"{tuple_delimiter}"strangers"{tuple_delimiter}"expresses gratitude to"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Author"{tuple_delimiter}"Allison Fragakis"{tuple_delimiter}"grateful for nutrition advice and recipe testing"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Author"{tuple_delimiter}"Susan Miller"{tuple_delimiter}"inspired by"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Author"{tuple_delimiter}"David Miller"{tuple_delimiter}"inspired by"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Author"{tuple_delimiter}"Sam Miller"{tuple_delimiter}"grateful for"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"friends"{tuple_delimiter}"indigenous diets"{tuple_delimiter}"helped explore benefits of"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"colleagues"{tuple_delimiter}"indigenous diets"{tuple_delimiter}"helped explore benefits of"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"strangers"{tuple_delimiter}"indigenous diets"{tuple_delimiter}"helped explore benefits of"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Ross Levy"{tuple_delimiter}"Author"{tuple_delimiter}"is the husband of"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Ross Levy"{tuple_delimiter}"project"{tuple_delimiter}"provided patience and support for"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Arlen Levy"{tuple_delimiter}"Author"{tuple_delimiter}"is the child of"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Arlen Levy"{tuple_delimiter}"recipes"{tuple_delimiter}"tasted and gave honest opinions on"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Emet Levy"{tuple_delimiter}"Author"{tuple_delimiter}"is the child of"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Emet Levy"{tuple_delimiter}"recipes"{tuple_delimiter}"tasted and gave honest opinions on"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Allison Fragakis"{tuple_delimiter}"project"{tuple_delimiter}"provided excellent nutrition advice to"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Allison Fragakis"{tuple_delimiter}"book"{tuple_delimiter}"spearheaded the recipe-testing portion of"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Susan Miller"{tuple_delimiter}"Author"{tuple_delimiter}"is the father of"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Susan Miller"{tuple_delimiter}"eating and cooking"{tuple_delimiter}"own wanderlust and love of"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"David Miller"{tuple_delimiter}"Author"{tuple_delimiter}"is the mother of"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"David Miller"{tuple_delimiter}"eating and cooking"{tuple_delimiter}"own wanderlust and love of"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Sam Miller"{tuple_delimiter}"Author"{tuple_delimiter}"is the brother of"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Sam Miller"{tuple_delimiter}"armchair nutrition"{tuple_delimiter}"is"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"friends"{tuple_delimiter}"friends"{tuple_delimiter}"contributors to project success"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"colleagues"{tuple_delimiter}"colleagues"{tuple_delimiter}"contributors to project success"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"strangers"{tuple_delimiter}"strangers"{tuple_delimiter}"contributors to project success"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Ross Levy"{tuple_delimiter}"Ross Levy"{tuple_delimiter}"beloved co-adventurer and supporter"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Arlen Levy"{tuple_delimiter}"Arlen Levy"{tuple_delimiter}"author's child who participated in recipe testing"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Emet Levy"{tuple_delimiter}"Emet Levy"{tuple_delimiter}"author's child who participated in recipe testing"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Allison Fragakis"{tuple_delimiter}"Allison Fragakis"{tuple_delimiter}"nutrition advisor and recipe tester"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Susan Miller"{tuple_delimiter}"Susan Miller"{tuple_delimiter}"author's parent with love of travel and food"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"David Miller"{tuple_delimiter}"David Miller"{tuple_delimiter}"author's parent with love of travel and food"{tuple_delimiter}){record_delimiter}
("relationship"{tuple_delimiter}"Author"{tuple_delimiter}"Author"{tuple_delimiter}"creator of the project and main figure of gratitude"{tuple_delimiter}){record_delimiter}
("content_keywords"{tuple_delimiter}"indigenous diets, gratitude, family, recipe testing, nutrition advice, inspiration, travel, cooking"{tuple_delimiter}){completion_delimiter}

-Real Data-

Text: {input_text}

Output:
"""

PROMPTS[
    "triples_continue_extraction"
] = """MANY relationships in the text were missed in the last extraction, add them below using the same format.
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