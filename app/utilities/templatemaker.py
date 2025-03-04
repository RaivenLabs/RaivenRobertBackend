from docxtpl import DocxTemplate
import json

class DocumentProcessor:
    def process_template(self, doc_path: str, variables: dict) -> DocxTemplate:
        """Process a Word template using docxtpl"""
        print("\n=== Starting Document Processing ===")
        print(f"Variables to replace: {json.dumps(variables, indent=2)}")
        
        # Load the template
        doc = DocxTemplate(doc_path)
        
        # Do the replacement
        doc.render(variables)
        
        print("\n=== Document Processing Complete ===")
        return doc
