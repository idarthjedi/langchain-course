from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()




def main():
    summary_template = """
    Given the information {information} about a person, I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    information = """
    John Calvin (/ˈkælvɪn/;[1] Middle French: Jehan Cauvin; French: Jean Calvin [ʒɑ̃ kalvɛ̃]; 10 July 1509 – 27 May 1564) was a French theologian, pastor and reformer in Geneva during the Protestant Reformation. He was a principal figure in the development of the system of Christian theology later called Calvinism, including its doctrines of predestination and of God's absolute sovereignty in the salvation of the human soul from death and eternal damnation. Calvinist doctrines were influenced by and elaborated upon Augustinian and other Christian traditions. Various Reformed Church movements, including Continental Reformed, Congregationalism, Presbyterianism, Waldensians, Baptist Reformed, Calvinist Methodism, and Reformed Anglican Churches, which look to Calvin as the chief expositor of their beliefs, have spread throughout the world.

    Calvin was a tireless polemicist and apologetic writer who generated much controversy. He also exchanged cordial and supportive letters with many reformers, including Philipp Melanchthon and Heinrich Bullinger. In addition to his seminal Institutes of the Christian Religion, Calvin wrote commentaries on most books of the Bible, confessional documents, and various other theological treatises.

    Calvin was originally trained as a humanist lawyer. He broke from the Roman Catholic Church around 1530. After religious tensions erupted in widespread deadly violence against Protestant Christians in France, Calvin fled to Basel, Switzerland, where in 1536 he published the first edition of the Institutes. In the same year, Calvin was recruited by Frenchman William Farel to join the Reformation in Geneva, where he regularly preached sermons throughout the week. However, the governing council of the city resisted the implementation of their ideas, and both men were expelled. At the invitation of Martin Bucer, Calvin proceeded to Strasbourg, where he became the minister of a church of French refugees. He continued to support the reform movement in Geneva, and in 1541 he was invited back to lead the church of the city."""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    llm = ChatOllama(temperature=0,
                     model="gpt-oss:20b")

    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})

    print(response.content)


if __name__ == "__main__":
    main()
