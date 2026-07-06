from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

document = """
The Eiffel Tower, located in Paris, France, was constructed between 1887 and 1889 as the entrance arch to the 1889 World’s Fair.

 It was designed by Gustave Eiffel’s engineering company and initially faced criticism from artists and intellectuals who considered it an eyesore.
 
 Today, the Eiffel Tower is one of the most visited monuments in the world, attracting over 7 million visitors annually.
 
 In 2015, special lighting systems were added to enhance its nighttime appearance and improve energy efficiency

"""

# splitting the document
splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)

chunks = splitter.split_text(document)
print("Document Chunks:\n")

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}")
    print(chunk)
    print("-" * 40)


class DocumentFacts(BaseModel):
    dates: list[str] = Field(description="Important dates")
    places: list[str]
    people: list[str]
    numbers: list[str]
    events: list[str]
    summary: str


parser = JsonOutputParser(pydantic_object=DocumentFacts)

prompt = PromptTemplate(
    template="""
    Extract the following from the document:

- Dates
- Places
- People
- Numbers
- Events

Then write a short summary.

{format_instructions}

Document:
{document}
    
    
    """,
    input_variables=["document"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

chain = prompt | llm | parser

result = chain.invoke({"document": document})

print("\n Extracted Info \n")
print(result)
print(result["summary"])


# ------    EXPECTED OUTPUT     ----------
# Document Chunks:

# Chunk 1
# The Eiffel Tower, located in Paris, France, was constructed between 1887 and 1889 as the entrance arch to the 1889
# ----------------------------------------
# Chunk 2
# arch to the 1889 World’s Fair.
# ----------------------------------------
# Chunk 3
# It was designed by Gustave Eiffel’s engineering company and initially faced criticism from artists and intellectuals
# ----------------------------------------
# Chunk 4
# and intellectuals who considered it an eyesore.
# ----------------------------------------
# Chunk 5
# Today, the Eiffel Tower is one of the most visited monuments in the world, attracting over 7 million visitors
# ----------------------------------------
# Chunk 6
# 7 million visitors annually.
# ----------------------------------------
# Chunk 7
# In 2015, special lighting systems were added to enhance its nighttime appearance and improve energy efficiency
# ----------------------------------------

#  Extracted Info

# {'dates': ['1887', '1889', '2015'], 'places': ['Paris', 'France', 'Eiffel Tower'], 'people': ['Gustave Eiffel'], 'numbers': ['7 million'], 'events': ['1889 World’s Fair'], 'summary': 'The Eiffel Tower was constructed between 1887 and 1889 as the entrance arch to the 1889 World’s Fair and has become one of the most visited monuments in the world.'}
# The Eiffel Tower was constructed between 1887 and 1889 as the entrance arch to the 1889 World’s Fair and has become one of the most visited monuments in the world.
