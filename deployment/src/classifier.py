import spacy
from spacy.tokens import Doc
from spacy.language import Language

Doc.set_extension("org_subtypes", default={}, force=True)

@Language.factory("org_subclassifier")
class OrganizationSubclassifier:
    """A component that filters for PERSON/ORG entities and subclassifies organizations"""
    
    def __init__(self, nlp, name):
        self.name = name
        
        # Keywords for university classification
        self.university_keywords = [
            "university", "college", "institute", "school", 
            "academy", "polytechnic", "conservatory"
        ]
        
        # Keywords for company classification
        self.company_keywords = [
            "inc", "corp", "ltd", "limited", "llc", "company", 
            "technologies", "systems", "group", "industries"
        ]
        
        # Known entities
        self.known_entities = {
            "mit": "UNIVERSITY",
            "harvard": "UNIVERSITY",
            "oxford": "UNIVERSITY",
            "cambridge": "UNIVERSITY",
            "apple": "COMPANY",
            "google": "COMPANY",
            "microsoft": "COMPANY",
            "amazon": "COMPANY"
        }
    
    def __call__(self, doc):
        # Filter entities to keep only PERSON and ORG
        filtered_ents = []
        
        # If no entities were found but we have text, try to classify it
        if not doc.ents and len(doc.text) > 0:
            # Since we're dealing with single entities, try to classify the whole text
            entity_text = doc.text.lower()
            entity_type = self._guess_entity_type(entity_text)
            
            if entity_type in ["PERSON", "ORG"]:
                # Create a span covering the whole text
                span = doc.char_span(0, len(doc.text), label=entity_type)
                if span:
                    filtered_ents.append(span)
        else:
            # Filter existing entities
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG"]:
                    filtered_ents.append(ent)
        
        # Overwrite doc.ents with our filtered list
        doc.ents = filtered_ents
        
        # Subclassify organizations
        for ent in doc.ents:
            if ent.label_ == "ORG":
                subtype = self._classify_organization(ent.text)
                doc._.org_subtypes[ent.text] = subtype
        
        return doc
    
    def _guess_entity_type(self, text):
        """Guess if an entity is a PERSON or ORG when spaCy NER doesn't detect it"""
        text = text.lower()
        
        # Check for known entities
        for entity, subtype in self.known_entities.items():
            if entity in text:
                return "ORG"  # All our known entities are organizations
        
        # Check for organization keywords
        if any(keyword in text for keyword in self.university_keywords + self.company_keywords):
            return "ORG"
        
        # Check for person-like patterns (1-3 words, no special chars except ' and -)
        words = text.split()
        if (1 <= len(words) <= 3 and 
            all(word.isalpha() or "'" in word or "-" in word for word in words)):
            return "PERSON"
        
        # Default to ORG 
        return "ORG"
    
    def _classify_organization(self, text):
        """Subclassify an organization as UNIVERSITY or COMPANY"""
        text = text.lower()
        
        # Check known entities first
        for entity, subtype in self.known_entities.items():
            if entity in text:
                return subtype
        
        # Check for university indicators
        if any(keyword in text for keyword in self.university_keywords) or " of " in text:
            return "UNIVERSITY"
        
        # Default to COMPANY
        return "COMPANY"
    
if __name__ == "__main__":
    OrganizationSubclassifier()