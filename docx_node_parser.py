from docx import Document
from docx.table import Table

from docx import Document
from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import _Cell, Table,_Row
from docx.text.paragraph import Paragraph

from typing import Any, Dict, List, Optional, Sequence
import logging
import json
from llama_index.core import SimpleDirectoryReader
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.utils import get_tqdm_iterable
from llama_index.core import Settings


_logger = logging.getLogger(__name__)

def parse_file(fileName,chunk_size):    
    def iter_block_items(parent):
        """
        Generate a reference to each paragraph and table child within *parent*,
        in document order. Each returned value is an instance of either Table or
        Paragraph. *parent* would most commonly be a reference to a main
        Document object, but also works for a _Cell object, which itself can
        contain paragraphs and tables.
        """
        if isinstance(parent, _Document):
            parent_elm = parent.element.body
        elif isinstance(parent, _Cell):
            parent_elm = parent._tc
        elif isinstance(parent, _Row):
            parent_elm = parent._tr
        else:
            raise ValueError("something's not right")
        for child in parent_elm.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, parent)
            elif isinstance(child, CT_Tbl):
                yield Table(child, parent)

    def convert2TextList(fileName,chunk_size):
        nodeList = []
        document = Document(fileName)
        for block in iter_block_items(document):                        
            if isinstance(block, Paragraph):
                if block.text == '':
                    continue
                # 处理超长文本，直接做截断处理
                blockNode = {'text':block.text if len(block.text)<=chunk_size else block.text[:chunk_size],'styleName':block.style.name}
                # if block.style.name not in ['Normal','Body Text']:
                #     print(block.style.name,block.text[:20])
                nodeList.append(blockNode)
            elif isinstance(block, Table):
                nodeText = ''
                # meta_keywords +=',表格,列表,清单'
                for row in block.rows:
                    row_data = []
                    for cell in row.cells:
                        for paragraph in cell.paragraphs:
                            row_data.append(paragraph.text)                    
                    nodeText+=r'\n'+r"\t".join(row_data)
                    if len(nodeText) >=chunk_size:
                        # 超长table，进行截断处理
                        blockNode = {'text':nodeText if len(nodeText)<=chunk_size else nodeText[:chunk_size],'styleName':'Table'}                        
                        nodeList.append(blockNode)
                        nodeText =''
                if nodeText != '':
                    blockNode = {'text':nodeText if len(nodeText)<=chunk_size else nodeText[:chunk_size],'styleName':'Table'}                        
                    nodeList.append(blockNode)
                    nodeText =''                    
        return nodeList
    def buildTree(textList,chunk_size,cutStyles)->dict:
        if len(textList) == 0:
            return  []
        nodeList = []      
        topNode = {}
        childs = []        
        for idx, item in enumerate(textList):                 
            if item['styleName'] in cutStyles:                  
                deepLevel =  cutStyles.index(item['styleName'])
                if deepLevel == 0:
                    if idx ==0:
                        # 首节点
                        topNode = item
                    else:
                        # 遇到新的切割符，前面的所有子节点递归处理
                        if len(cutStyles) >1 and len(childs)>1:
                            childs = buildTree(childs,chunk_size,cutStyles[1:])
                        if len(childs) >=1:    
                            topNode['childs'] = childs                           
                        nodeList.append(topNode)
                        topNode = item
                        childs = []
                else:
                    # 子节点，                                     
                    childs.append(item)

            else:
                # 非分隔符,作为叶子节点
                childs.append(item)
        # 循环结束
        if len(cutStyles) >1 and len(childs)>1:
            childs = buildTree(childs,chunk_size,cutStyles[1:])
        if len(childs) >=1: 
            topNode['childs'] = childs 
        if ('text' in topNode.keys()) or ('childs' in topNode.keys()):
            nodeList.append(topNode)
        return nodeList
    # 返回值：切片列表，未切片列表
    def cutTreeOnce(textTree,chunk_size) :
        def caculateCharCount(node):            
            cnt =0
            if node ==None:
                return cnt
            if ('text' in node.keys()):
                cnt= len(node['text'])
            if ('childs' in node.keys()):
                for item in node['childs']:
                    cnt+=caculateCharCount(item)
            return cnt
        sumCnt = 0        
        for idx,item in enumerate(textTree):
            cnt = caculateCharCount(item)
            if sumCnt+cnt > chunk_size*1.2:
                if idx ==0:
                    # 有子节点进一步切分
                    if ('childs' in item.keys()):
                        cutedTree,unCutTree= cutTreeOnce(item['childs'],chunk_size)
                        # item被拆分为两部分:已切片、未切片
                        newCutedNode = item.copy()
                        newCutedNode['childs'] = cutedTree

                        # newUnCutedNode = []
                        # if not (unCutedTree == None or len(unCutedTree) ==0)
                        unCutedTree = []
                        if len(unCutTree) >0:
                            newUnCutedNode = item.copy()
                            newUnCutedNode['childs'] =unCutTree                        
                            unCutedTree = [newUnCutedNode]                        
                        if len(textTree) >1:
                            # 如果后面还有节点，则拼接
                            unCutedTree.extend(textTree[1:])
                        return [newCutedNode],unCutedTree
                    else:
                        # 无子节点，直接返回
                        if len(textTree) >1:
                            return [item],textTree[1:]
                        else:
                            return [item],[]
                else:
                    # 完成切片
                    return textTree[0:idx],textTree[idx:]                        
            else:
                sumCnt +=cnt
        return textTree,[]
    def node2str(node):
        retStr = ''
        if ('text' in node.keys()):
            retStr= node['text']
        if ('childs' in node.keys()):
            for item in node['childs']:
                retStr += r'\n'+node2str(item)
        return retStr

    allCutStyles = ['Heading 1','Heading 2','Heading 3','Heading 4','Heading 5','Table','Normal']
    # 获取文本块列表
    textList = convert2TextList(fileName,chunk_size)
    #构建结构树
    textTree = buildTree(textList,chunk_size,allCutStyles)
    with open('./temp/1.json',mode='w') as file:
        json.dump(textTree,file,indent=4)
        file.close()
    # 切片处理
    toCutTree = textTree
    # 构建返回列表
    nodeList =[]
    idx =0
    while len(toCutTree) >0:
        cutedTree,toCutTree = cutTreeOnce(toCutTree,chunk_size)
        cutedStr =''
        for item in cutedTree:
           cutedStr += node2str(item)+r'\n'
        nodeList.append((cutedStr,{}))
        # print(str(idx),':',cutedStr)
        idx +=1
    # print('切片完成')
    # print(nodeList)
    return nodeList


class docxNodeParser(NodeParser):
    """Custom Markdown node parser.

    Splits a document into Nodes using custom Markdown splitting logic to split based on heading levels.

    Args:
        include_metadata (bool): whether to include metadata in nodes
        heading_level (int): level of heading to split on

    """

    @classmethod
    def from_defaults(
        cls,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "docxNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        return cls(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "docxNodeParser"

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            nodes = self.get_nodes_from_node(node, **kwargs)
            all_nodes.extend(nodes)

        return all_nodes

    def _headings_processor(
        self, segments: list, headings: list, current_heading: str, level: int
    ) -> List[str]:
        # add back the headings if they were split on

        splits = []
        # heading level corresponds to #, ##, ###, etc.
        heading_level = level + 1

        for segment in segments:
            segment_has_parent = [
                segment.startswith(parent_heading)
                for parent_heading in headings[0:heading_level]
            ]

            if True in segment_has_parent:
                for parent_heading in headings[0:heading_level]:
                    if segment.startswith(parent_heading):
                        splits.append(segment)

            else:
                splits.append(current_heading + segment)

        return splits

    def _document_splitter(
        self,
        heading: str,
        document: list,
        heading_level: int,
        headings: list = ["\n# ", "\n## ", "\n### ", "\n#### "],
    ) -> List[str]:
        documents = []
        split_docs = []

        for doc in document:

            if heading not in doc:
                # If the heading is not at the current level, then don't process the doc
                _logger.debug(f"skipping doc {doc[0:100]}")
                split_docs.append(doc)

            else:
                _logger.debug(f"processing doc:  {doc[0:100]}")
                segments = doc.split(heading)

                # if the heading doesn't start with a previous heading then add current heading
                splits = self._headings_processor(
                    segments, headings, heading, heading_level
                )

                split_docs.extend(splits)

            documents.extend(split_docs)
            split_docs = []

        return documents

    def get_nodes_from_node(self, node: BaseNode, **kwargs) -> List[TextNode]:
        """Get Nodes from document basedon headers"""
        fileName = node.metadata['file_path']
        # print(fileName)
        markdown_nodes = []
        textNodes = parse_file(fileName=fileName,chunk_size= Settings.chunk_size)
        for text,metadata in textNodes:
            markdown_nodes.append(self._build_node_from_split(text, node,metadata))
        # text = node.get_content(metadata_mode=MetadataMode.NONE)
        # markdown_nodes = []

        # # heading level can get passed as kwargs
        # headings = self._split_on_heading(text, **kwargs)
        # headings_w_metadata = self._get_heading_text(headings, **kwargs)

        # for heading, metadata in headings_w_metadata:
        #     markdown_nodes.append(self._build_node_from_split(heading, node, metadata))

        return markdown_nodes

    def _build_node_from_split(
        self,
        text_split: str,
        node: BaseNode,
        metadata: dict,
    ) -> TextNode:
        """Build node from single text split."""

        textNode = build_nodes_from_splits([text_split], node, id_func=self.id_func)[0]

        if self.include_metadata:
            textNode.metadata = {**node.metadata, **metadata}

        return textNode


# if __name__ == "__main__":

#     from llama_index.readers.file import FlatReader
#     from pathlib import Path

#     # md_docs = FlatReader().load_data(Path("example_source.md"))
#     documents = SimpleDirectoryReader("./data",recursive=False).load_data()

#     # print(md_docs)
#     # print(len(md_docs))

#     parser = docxNodeParser()
#     nodes = parser.get_nodes_from_documents(documents, heading_level=2)

#     print(nodes)
#     print(len(nodes))
