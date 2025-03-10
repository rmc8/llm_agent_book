from langchain_core.documents import Document
from langchain_chroma import Chroma

document = Document(
    page_content="""\
    セダムはベンケイソウ科マンネングサ属で、日本にも自生しているポピュラーな多肉植物です。
    種類が多くて葉の大きさや形状、カラーバリエーションも豊富なので、組み合わせて寄せ植えにしたり、
    庭のグランドカバーにしたりして楽しむことがｄけいます。とても丈夫で育てやすく、多肉植物を初めて育てる方にもおすすめです。
    """,
    metadata={"source": "succulent-plants-doc"},
)
print(document)
