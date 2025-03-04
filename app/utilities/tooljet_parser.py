# tooljet_parser.py
from typing import Dict, List, Any
import re

class TooljetParser:
    def __init__(self):
        self.component_mappings = {
            'form': {
                'type': 'form',
                'properties': {
                    'showHeader': True,
                    'submitButtonText': 'Submit',
                }
            },
            'table': {
                'type': 'table',
                'properties': {
                    'enablePagination': True,
                    'enableSearch': True,
                }
            },
            'calendar': {
                'type': 'calendar',
                'properties': {
                    'enableDateSelection': True,
                }
            },
            'kanban': {
                'type': 'kanban',
                'properties': {
                    'enableDragAndDrop': True,
                }
            },
            'modal': {
                'type': 'modal',
                'properties': {
                    'closeOnClickOutside': True,
                }
            },
            'richText': {
                'type': 'richText',
                'properties': {
                    'enableFormatting': True,
                }
            },
            'progressBar': {
                'type': 'progressBar',
                'properties': {
                    'showPercentage': True,
                }
            }
        }

    def parse_workflow_text(self, workflow_text: str) -> Dict[str, Any]:
        """Parse the workflow text into Tooljet canvas JSON"""
        steps = self._extract_steps(workflow_text)
        return self._generate_canvas_json(steps)

    def _extract_steps(self, workflow_text: str) -> List[Dict[str, Any]]:
        """Extract structured step information from workflow text"""
        steps = []
        current_step = None
        
        for line in workflow_text.split('\n'):
            line = line.strip()
            
            # New step
            if line.startswith('Step'):
                if current_step:
                    steps.append(current_step)
                current_step = {
                    'name': line,
                    'components': [],
                    'data_requirements': [],
                    'user_flow': []
                }
            
            # Components section
            elif 'Components:' in line:
                component_section = True
            elif line.startswith('- ') and component_section:
                component = self._parse_component_line(line)
                if component:
                    current_step['components'].append(component)

        if current_step:
            steps.append(current_step)
            
        return steps

    def _parse_component_line(self, line: str) -> Dict[str, Any]:
        """Parse a component line into structured data"""
        # Remove the bullet point and split by colon
        line = line.replace('- ', '')
        parts = line.split(': ', 1)
        
        if len(parts) != 2:
            return None
            
        component_type, purpose = parts
        component_type = component_type.lower()
        
        # Map to Tooljet component type
        if component_type in self.component_mappings:
            return {
                **self.component_mappings[component_type],
                'purpose': purpose
            }
            
        return None

    def _generate_canvas_json(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate Tooljet canvas JSON from parsed steps"""
        canvas = {
            'version': 1,
            'components': {},
            'dataQueries': [],
            'layouts': {
                'desktop': {
                    'rows': []
                }
            }
        }
        
        current_y = 0
        
        for step in steps:
            # Add a section header
            canvas['components'][f"header_{step['name']}"] = {
                'type': 'text',
                'properties': {
                    'text': step['name'],
                    'textSize': 'lg',
                    'textColor': '#1f2937'
                },
                'layout': {
                    'top': current_y,
                    'left': 0,
                    'width': 12  # Full width in grid units
                }
            }
            current_y += 50
            
            # Add components for this step
            for idx, component in enumerate(step['components']):
                component_id = f"{step['name']}_{component['type']}_{idx}"
                canvas['components'][component_id] = {
                    'type': component['type'],
                    'properties': {
                        **component['properties'],
                        'title': component['purpose']
                    },
                    'layout': {
                        'top': current_y,
                        'left': 0,
                        'width': 12  # Full width in grid units
                    }
                }
                current_y += 200  # Spacing between components
                
        return canvas

    def _create_data_queries(self, data_requirements: List[str]) -> List[Dict[str, Any]]:
        """Create Tooljet data queries from data requirements"""
        queries = []
        for requirement in data_requirements:
            # Convert requirement into appropriate data query
            # This would be customized based on your data sources
            pass
        return queries
