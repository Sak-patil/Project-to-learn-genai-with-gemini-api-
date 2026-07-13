import asyncio
from mcp import ClientSession       #this represent the connection with mcp server 
from mcp.client.stdio import stdio_client  #as there are many protocols by which the client and sever can talk to each other
                                            #but as our client and server are on our laptop we are going to use the stdinput and stdoutput protocol 
from mcp import StdioServerParameters
import os
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types

async def main():

    # ----------------------------
    # Start MCP Server
    # ----------------------------

    server_params = StdioServerParameters(
        command="python",
        args=["server.py"]
    )

    async with stdio_client(server_params) as (read_stream, write_stream):

        async with ClientSession(read_stream, write_stream) as session:

            await session.initialize()

            # ----------------------------
            # Discover MCP Tools
            # ----------------------------

            mcp_tools = await session.list_tools()

            # Convert MCP tools to Gemini FunctionDeclaration
            gemini_tools = []

            for tool in mcp_tools.tools:

                gemini_tools.append(
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.inputSchema
                    )
                )

            tool = types.Tool(
                function_declarations=gemini_tools
            )

            client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY")
            )

            messages = []

            while True:

                user_input = input("You : ")

                if user_input.lower() in ["exit", "quit"]:
                    break

                messages.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=user_input)]
                    )
                )

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=messages,
                    config=types.GenerateContentConfig(
                        tools=[tool]
                    )
                )

                model_message = response.candidates[0].content

                messages.append(model_message)

                tool_called = False

                for part in model_message.parts:
                                    if part.function_call:
                                        print("Gemini requested tool:", part.function_call.name)
                                        print("Arguments:", dict(part.function_call.args))

                for part in model_message.parts:

                    if part.function_call:

                        tool_called = True

                        tool_name = part.function_call.name

                        args = dict(part.function_call.args)

                        result = await session.call_tool(
                            tool_name,
                            args
                        )

                        print(result.model_dump())
                        messages.append(
                            types.Content(
                                role="tool",
                                parts=[
                                    types.Part.from_function_response(
                                        name=tool_name,
                                        response=result.model_dump()
                                    )
                                ]
                            )
                        )

                if tool_called:

                    final_response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=messages,
                        config=types.GenerateContentConfig(
                            tools=[tool]
                        )
                    )

                    print(final_response.text)

                    messages.append(
                        types.Content(
                            role="model",
                            parts=[
                                types.Part(text=final_response.text)
                            ]
                        )
                    )

                else:

                    print(response.text)


if __name__ == "__main__":
    asyncio.run(main())