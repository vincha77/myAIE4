"""
pdf_to_text.py
This class loads a list of pdf documents passed in 
and returns a list of parsed text for these docs
"""

# importing required classes
import os
from pypdf import PdfReader 

class TextFromPyPdf:
    '''
    this class converts a list of pdf documents into a list of text documents
    '''
    def __init__(self,
                 list_of_pdf_docs=None):
        if isinstance(list_of_pdf_docs, list) and len(list_of_pdf_docs) > 0:
            self.list_of_pdf_docs = list_of_pdf_docs
            # print(f'initialized Class with a list of {len(self.list_of_pdf_docs)} pdf documents ')
        else:
            print('ERROR: expecting a non-empty list of pdf names to be passed in')
            raise Exception
        return
    
    def process_single_pdf(self, pdfdoc):
        # check if file exists; if not return None
        if os.path.isfile(pdfdoc):
            pass
        else:
            print(f'Warning: pdf file {pdfdoc} does not exist...skipping to next pdf file')
            return None
        reader = PdfReader(pdfdoc)
        numpages = len(reader.pages)
        thistext = ''
        for pagecount in range(0, numpages):
            page = reader.pages[pagecount]
            pagetext = page.extract_text()
            thistext = thistext + ' ' + pagetext  # adding a line break
            # print('\n')
            # print(thistext)
        return thistext
    
    def process_all_pdfs(self):
        list_of_texts = []
        for pdfdoc in self.list_of_pdf_docs:
            pdftext = self.process_single_pdf(pdfdoc)
            if pdftext is not None:
                list_of_texts.append([pdftext])
        return list_of_texts


# This section tests the code with mock data - two tiny pdfs!!!
if __name__ == "__main__":
    # list_of_pdfs = ['../data/pdfOne.pdf']
    # list_of_pdfs = ['../data/pdfOne.pdf',
    #                 '../data/wrongfile.pdf']
    # list_of_pdfs = 'abcd'

    list_of_pdfs = ['../data/pdfOne.pdf',
                    '../data/pdfTwo.pdf']
    
    textfrompdf = TextFromPyPdf(list_of_pdf_docs=list_of_pdfs)
    list_of_documents = textfrompdf.process_all_pdfs()

    print(list_of_documents)
    print(list_of_documents[0])
    print(list_of_documents[0][0])
    print(f'number of documents returned is: {len(list_of_documents)} ')
