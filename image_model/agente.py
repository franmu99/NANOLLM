import networkx as nx
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferMemory
from typing import List, Union
import re
import torch
from PIL import Image
from image_model.model import ImageAttentionModel
from torchvision import transforms
import openai

# Configurar la API key de OpenAI
openai.api_key = 'tu_api_key_de_openai'

# Cargar tu modelo de generación de imágenes
model = ImageAttentionModel(d_model=64, num_heads=8, num_layers=4)
model.load_state_dict(torch.load("image_attention_model.pth"))
model.eval()

def process_images(image_paths):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    processed_images = []
    for path in image_paths:
        image = Image.open(path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output, _ = model(image_tensor)
        # Aquí debes implementar la lógica para interpretar la salida de tu modelo
        # y convertirla en una descripción textual
        processed_images.append(f"Descripción de la imagen {path}")
    
    return processed_images

# Herramienta para procesar imágenes
image_tool = Tool(
    name="Image Analyzer",
    func=process_images,
    description="Útil para analizar y describir múltiples imágenes."
)

def analyze_description(description):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Analiza la siguiente descripción de imagen y extrae los elementos clave para la generación:\n\n{description}",
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Herramienta para analizar descripciones
description_tool = Tool(
    name="Description Analyzer",
    func=analyze_description,
    description="Útil para analizar y extraer elementos clave de la descripción del usuario."
)

def generate_image(description, analysis, image_descriptions, memory):
    # Recuperar el historial de feedback de la memoria
    feedback_history = memory.chat_memory.messages

    combined_input = f"Descripción: {description}\nAnálisis: {analysis}\nImágenes de referencia: {', '.join(image_descriptions)}"
    combined_input += f"\nHistorial de feedback: {feedback_history}"
    
    # Aquí debes implementar la lógica para generar una nueva imagen
    # basada en la entrada combinada usando tu modelo
    # Considera el historial de feedback para refinar la imagen
    
    return "URL de la imagen generada basada en la descripción, análisis, referencias y historial de feedback"

# Herramienta para generar imágenes
generation_tool = Tool(
    name="Image Generator",
    func=generate_image,
    description="Útil para generar una nueva imagen basada en una descripción, análisis y referencias."
)

# Nuevo módulo de supervisión
def supervise_image(image_url, description, analysis):
    # Aquí implementarías la lógica para evaluar la calidad de la imagen generada
    # Esta función podría utilizar otro modelo de IA o un conjunto de reglas predefinidas
    
    # Por ahora, simularemos una evaluación básica
    quality_score = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Evalúa la calidad de este resultado (escala 1-10) basado en la siguiente descripción y análisis:\n\nDescripción: {description}\nAnálisis: {analysis}\nResultado: {action_result}\n\nPuntuación:",
        max_tokens=1,
        n=1,
        stop=None,
        temperature=0.5,
    ).choices[0].text.strip()
    
    try:
        quality_score = int(quality_score)
    except ValueError:
        quality_score = 5  # valor por defecto si la conversión falla
    
    feedback = ""
    if quality_score < 7:
        feedback = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"Proporciona sugerencias para mejorar este resultado:\n\nDescripción: {description}\nAnálisis: {analysis}\nResultado: {action_result}\n\nSugerencias:",
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.7,
        ).choices[0].text.strip()
    
    return quality_score, feedback

class SupervisedTool(Tool):
    def __init__(self, name, func, description, supervisor_func):
        super().__init__(name=name, func=func, description=description)
        self.supervisor_func = supervisor_func

    def run(self, tool_input, description, analysis):
        result = self.func(tool_input)
        quality_score, feedback = self.supervisor_func(result, description, analysis)
        return {
            "result": result,
            "quality_score": quality_score,
            "feedback": feedback
        }

# Crear herramientas supervisadas
description_tool = SupervisedTool(
    name="Description Analyzer",
    func=analyze_description,
    description="Útil para analizar y extraer elementos clave de la descripción del usuario.",
    supervisor_func=supervise_action
)

image_tool = SupervisedTool(
    name="Image Analyzer",
    func=process_images,
    description="Útil para analizar y describir múltiples imágenes.",
    supervisor_func=supervise_action
)

generation_tool = SupervisedTool(
    name="Image Generator",
    func=generate_image,
    description="Útil para generar una nueva imagen basada en una descripción, análisis y referencias.",
    supervisor_func=supervise_action
)

# Actualizar la plantilla de prompt
template = """Eres un asistente creativo que puede analizar descripciones, procesar imágenes y generar nuevas imágenes.
Cada acción que realices será supervisada para garantizar la calidad.
Tienes acceso a las siguientes herramientas:

{tools}

Usa el siguiente formato:

Pregunta: la pregunta del usuario
Pensamiento: siempre piensa en qué hacer
Acción: la acción a tomar, debe ser una de [{tool_names}]
Entrada de acción: la entrada para la acción
Observación: el resultado de la acción, incluyendo la puntuación de calidad y el feedback
... (este Pensamiento/Acción/Entrada de acción/Observación puede repetirse N veces)
Pensamiento: Ahora sé la respuesta final
Respuesta final: la respuesta final a la pregunta del usuario

Historial de la conversación:
{history}

Si recibes feedback sobre una acción, úsalo para mejorar en la siguiente iteración.
Aprende de los errores pasados y evita repetirlos.

Pregunta: {input}
{agent_scratchpad}"""

# Implementar el parseador de salida del agente
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Respuesta final:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Respuesta final:")[-1].strip()},
                log=llm_output,
            )
        regex = r"Acción: (.*?)[\n]*Entrada de acción: (.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

# Configurar la memoria
memory = ConversationBufferMemory(memory_key="history")

# Configurar el agente principal
llm = OpenAI(temperature=0.7)
prompt = StringPromptTemplate(
    template=template,
    tools=[description_tool, image_tool, generation_tool],
    input_variables=["input", "history", "agent_scratchpad"]
)
output_parser = CustomOutputParser()

llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
tool_names = [tool.name for tool in [description_tool, image_tool, generation_tool]]
main_agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

# Crear el grafo de agentes
agent_graph = nx.DiGraph()
agent_graph.add_node("Main Agent", agent=main_agent)
agent_graph.add_node("Description Analyzer", agent=description_tool)
agent_graph.add_node("Image Analyzer", agent=image_tool)
agent_graph.add_node("Image Generator", agent=generation_tool)

agent_graph.add_edge("Main Agent", "Description Analyzer")
agent_graph.add_edge("Main Agent", "Image Analyzer")
agent_graph.add_edge("Main Agent", "Image Generator")

class SupervisedAgentExecutor(AgentExecutor):
    def _execute_agent_action(self, agent_action: AgentAction) -> str:
        tool = self._get_tool(agent_action.tool)
        observation = tool.run(
            agent_action.tool_input,
            description=self.input_description,
            analysis=self.input_analysis
        )
        return f"Resultado: {observation['result']}\nPuntuación de calidad: {observation['quality_score']}/10\nFeedback: {observation['feedback']}"

def handle_user_input(user_input, image_paths):
    agent_executor = SupervisedAgentExecutor.from_agent_and_tools(
        agent=main_agent, 
        tools=[description_tool, image_tool, generation_tool], 
        memory=memory,
        verbose=True
    )
    
    agent_executor.input_description = user_input
    agent_executor.input_analysis = analyze_description(user_input)
    
    while True:
        result = agent_executor.run(
            input=f"Descripción del usuario: {user_input}\n"
                  f"Imágenes de referencia: {len(image_paths)} imágenes proporcionadas\n"
                  "Analiza la descripción, procesa las imágenes de referencia y genera una nueva imagen basada en esta información."
        )
        
        print("Chatbot:", result)
        
        user_feedback = input("¿Estás satisfecho con este resultado? (sí/no): ").lower()
        if user_feedback == 'sí' or user_feedback == 'si':
            memory.save_context({"input": "Resultado final"}, {"output": "Usuario satisfecho con el resultado"})
            return result
        else:
            feedback = input("Por favor, describe qué aspectos te gustaría mejorar: ")
            memory.save_context({"input": "Feedback del usuario"}, {"output": feedback})
            print("Entendido. Voy a intentar mejorar el resultado teniendo en cuenta tu feedback.")

# Bucle principal del chatbot
if __name__ == "__main__":
    while True:
        user_input = input("Describe la imagen que quieres generar: ")
        if user_input.lower() == "salir":
            break
        image_paths = input("Introduce las rutas de las imágenes de referencia separadas por comas: ").split(',')
        final_result = handle_user_input(user_input, image_paths)
        print("Chatbot: La imagen final generada está disponible aquí:", final_result)
        
        