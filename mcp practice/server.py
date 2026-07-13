from mcp.server.fastmcp import FastMCP #It is the class that helps us create an MCP server
mcp = FastMCP("Calculator Server")#creatibng object 
from datetime import datetime

@mcp.tool(description="Calculates a mathematical expression.")   #its like making th list it registers the tools in list 
def calculator(expression: str):
    return eval(expression)


@mcp.tool(description="Returns the current day of the week.")
def current_day():
    return datetime.now().strftime("%A")

if __name__ == "__main__":
    mcp.run()            #It starts the server and waits.It will keep running until stop it (Ctrl + C).