#!/usr/bin/env python
"""
Template Converter Utility
-------------------------
Converts source templates to Jinja2-compatible templates using Word COM automation
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Import Win32 libraries - must be installed on the server
try:
    import win32com.client as win32
    import pythoncom
except ImportError:
    print("Error: win32com is required. Install with: pip install pywin32", file=sys.stderr)
    sys.exit(1)

# Determine the script directory and add it to the path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Try to import variables_config from the same directory
try:
    from variables_config import (
        provider_terms, customer_terms, effective_date_terms,
        other_terms, other_terms_also, yet_more_other_terms,
        exact_replacements as base_exact_replacements,
        wildcard_replacements as base_wildcard_replacements,
        cleanup_replacements as base_cleanup_replacements,
        add_variations
    )
except ImportError:
    print("Warning: variables_config.py not found in script directory. "
          "Using default configurations.", file=sys.stderr)
    # Fallback to default configurations if import fails
    provider_terms = ['Supplier', 'Provider', 'Vendor', 'Contractor', 'Consultant']
    customer_terms = ['Company', 'Client', 'Customer', 'Purchaser', 'Buyer', 'Service Provider']
    effective_date_terms = ['Order Effective Date', 'Effective Date', 'Agreement Effective Date', 
                          'Contract Effective Date']
    other_terms = []
    other_terms_also = []
    yet_more_other_terms = []
    
    def add_variations(terms):
        variations = []
        for term in terms:
            variations.append(f'("{term}")')
            variations.append(f'(the "{term}")')
        return variations
    
    base_exact_replacements = {
        'Authorized Buyer': '{{CustomerDefinedTerm}}',
    }
    
    base_wildcard_replacements = {
        '_@ *Kk979kK': ' {{EffectiveDate}}',
        '_@ *Kk97999kK': ' {{ContractorDefinedTerm}}',
        '_@ *Kk97099kK': ' {{CustomerDefinedTerm}}',
        '_@ *Kk961kK': ' {{OtherTerms}}',
        '_@ *Kk961999kK': ' {{MoreTerms}}',
        '_@ *Kk961099kK': ' {{YetMoreTerms}}'
    }
    
    base_cleanup_replacements = {}

# Default paths - will be overridden by environment variables or arguments
DEFAULT_SOURCE_PATH = os.path.join(SCRIPT_DIR, '..', '..','..', 'static', 'data', 'SystemITTemplates', 
                                 'TangibleITTemplates', 'Source')
DEFAULT_FOUNDATIONAL_PATH = os.path.join(SCRIPT_DIR, '..', '..','..', 'static', 'data', 'SystemITTemplates', 
                                       'TangibleITTemplates', 'Foundational')
DEFAULT_ARCHIVE_PATH = os.path.join(SCRIPT_DIR, '..', '..', '..','static', 'data', 'SystemITTemplates',
                                  'TangibleITTemplates', 'ArchivedSource')

# Get base paths from environment if available
SOURCE_BASE_PATH = os.environ.get('TEMPLATE_SOURCE_PATH', DEFAULT_SOURCE_PATH)
FOUNDATIONAL_BASE_PATH = os.environ.get('TEMPLATE_FOUNDATIONAL_PATH', DEFAULT_FOUNDATIONAL_PATH)
ARCHIVE_BASE_PATH = os.environ.get('TEMPLATE_ARCHIVE_PATH', DEFAULT_ARCHIVE_PATH)

def log_message(message):
    """Print a message to stdout for status tracking"""
    print(message, flush=True)

def log_error(message):
    """Print an error message to stderr"""
    print(message, file=sys.stderr, flush=True)

def cleanup_word():
    """Force cleanup of any existing Word processes"""
    try:
        import subprocess
        subprocess.run(['taskkill', '/IM', 'winword.exe', '/F'], 
                      capture_output=True, 
                      check=False)
        time.sleep(2)
    except Exception as e:
        log_error(f"Cleanup warning: {str(e)}")

def safe_find_replace(doc, find_text, replace_text, use_wildcards=False):
    """
    Perform a single find/replace operation with safety checks
    """
    log_message(f"Replacing: '{find_text}' with '{replace_text}'")
    
    # First do the main content
    find = doc.Content.Find
    find.ClearFormatting()
    find.Replacement.ClearFormatting()
    
    # Basic setup
    find.Text = find_text
    find.Replacement.Text = replace_text
    find.Forward = True
    find.Wrap = 1  # wdFindContinue
    find.Format = False
    find.MatchCase = True
    find.MatchWholeWord = not use_wildcards
    find.MatchWildcards = use_wildcards
    
    try:
        # Execute find/replace for main content
        find.Execute(
            FindText=find_text,
            ReplaceWith=replace_text,
            Replace=2,  # wdReplaceAll
            Forward=True,
            MatchCase=True,
            MatchWholeWord=not use_wildcards
        )
        
        # Now check footers
        for section in doc.Sections:
            for footer in section.Footers:
                if footer.Exists:
                    footer_find = footer.Range.Find
                    footer_find.ClearFormatting()
                    footer_find.Replacement.ClearFormatting()
                    footer_find.Text = find_text
                    footer_find.Replacement.Text = replace_text
                    footer_find.Forward = True
                    footer_find.Wrap = 1
                    footer_find.Format = False
                    footer_find.MatchCase = True
                    footer_find.MatchWholeWord = not use_wildcards
                    footer_find.MatchWildcards = use_wildcards
                    
                    footer_find.Execute(
                        FindText=find_text,
                        ReplaceWith=replace_text,
                        Replace=2,
                        Forward=True,
                        MatchCase=True,
                        MatchWholeWord=not use_wildcards
                    )
        
        return True
        
    except Exception as e:
        log_error(f"Warning: Could not replace '{find_text}': {str(e)}")
        return False

def build_replacements():
    """Build all replacement dictionaries using configured terms"""
    # Start with the base exact replacements
    exact_replacements = base_exact_replacements.copy()
    
    # Add provider terms
    for term in add_variations(provider_terms):
        exact_replacements[term] = f'Kk97999kK {term}'
    
    # Add company terms
    for term in add_variations(customer_terms):
        exact_replacements[term] = f'Kk97099kK {term}'
    
    # Add date terms
    for term in add_variations(effective_date_terms):
        exact_replacements[term] = f'Kk979kK {term}'
        
    # Add other term categories
    for term in add_variations(other_terms):
        exact_replacements[term] = f'Kk961kK {term}'
    
    for term in add_variations(other_terms_also):
        exact_replacements[term] = f'Kk961999kK {term}'
    
    for term in add_variations(yet_more_other_terms):
        exact_replacements[term] = f'Kk961099kK {term}'
    
    # Use the base wildcard and cleanup replacements
    wildcard_replacements = base_wildcard_replacements.copy()
    cleanup_replacements = base_cleanup_replacements.copy()
    
    return exact_replacements, wildcard_replacements, cleanup_replacements

def get_output_paths(template_path):
    """
    Generate output paths for processed template
    Returns: (temporary_path, processed_path, foundational_path, archive_path)
    """
    template_path = os.path.abspath(template_path)
    
    # Get the original filename and directory
    original_dir = os.path.dirname(template_path)
    original_filename = os.path.basename(template_path)
    file_stem = Path(template_path).stem
    
    # Temporary path in system temp directory
    temp_dir = Path(os.environ.get('TEMP', '/tmp')) / 'doc_processing'
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / f"temp_processed_{int(time.time())}.docx"
    
    # Processed path in same directory
    processed_path = os.path.join(original_dir, f"{file_stem}_processed.docx")
    
    # For both Foundational and Archive, determine the appropriate subdirectory
    if SOURCE_BASE_PATH in template_path:
        # Extract the subdirectory structure after SOURCE_BASE_PATH
        rel_path = os.path.relpath(original_dir, SOURCE_BASE_PATH)
        foundational_subdir = os.path.join(FOUNDATIONAL_BASE_PATH, rel_path)
        archive_subdir = os.path.join(ARCHIVE_BASE_PATH, rel_path)
    else:
        # If not in SOURCE_BASE_PATH, use the same directory name
        dir_name = os.path.basename(original_dir)
        foundational_subdir = os.path.join(FOUNDATIONAL_BASE_PATH, dir_name)
        archive_subdir = os.path.join(ARCHIVE_BASE_PATH, dir_name)
    
    # Ensure the subdirectories exist
    os.makedirs(foundational_subdir, exist_ok=True)
    os.makedirs(archive_subdir, exist_ok=True)
    
    # Path for saving in Foundational with original filename
    foundational_path = os.path.join(foundational_subdir, original_filename)
    
    # Path for archiving the source file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive_filename = f"{file_stem}_{timestamp}.docx"
    archive_path = os.path.join(archive_subdir, archive_filename)
    
    return str(temp_path), processed_path, foundational_path, archive_path

def archive_source_file(file_path, archive_path):
    """Archive the original source file"""
    try:
        log_message(f"Archiving source file to: {archive_path}")
        shutil.copy2(file_path, archive_path)
        return True
    except Exception as e:
        log_error(f"Warning: Could not archive source file: {str(e)}")
        return False

def convert_template(source_full_path, target_path):
    word = None
    doc = None
    
    try:
        # Kill any existing Word processes first
        cleanup_word()
        time.sleep(2)  # Give more time for cleanup
        
        pythoncom.CoInitialize()
        
        word = win32.gencache.EnsureDispatch('Word.Application')
        word.Visible = False
        # Add more error handling for Word
        if not word:
            raise Exception("Failed to initialize Word application")
            
        temp_dir = Path(os.environ.get('TEMP', '/tmp')) / 'doc_processing'
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"temp_{int(time.time())}.docx"
        
        # Copy to temp and verify
        shutil.copy2(source_full_path, temp_path)
        if not os.path.exists(temp_path):
            raise Exception("Failed to create temporary file")
            
        # Open with retry
        for attempt in range(3):
            try:
                doc = word.Documents.Open(str(temp_path))
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(1)
                
        exact_replacements, wildcard_replacements, cleanup_replacements = build_replacements()
        
        # Do replacements with better error handling
        for replacements, use_wildcards in [
            (exact_replacements, False),
            (wildcard_replacements, True),
            (cleanup_replacements, False)
        ]:
            for find_text, replace_text in replacements.items():
                try:
                    safe_find_replace(doc, find_text, replace_text, use_wildcards)
                except Exception as e:
                    print(f"Warning: Failed to replace '{find_text}': {str(e)}")
                    # Continue with next replacement
        
        # Save with retry
        for attempt in range(3):
            try:
                doc.SaveAs2(target_path)
                log_message(f"✅ Verified file saved to: {target_path}")
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(1)
        
        return target_path
        
    finally:
        if doc:
            try:
                doc.Close()
            except:
                pass
        if word:
            try:
                word.Quit()
            except:
                pass
            finally:
                cleanup_word()
        try:
            pythoncom.CoUninitialize()
        except:
            pass

def write_status_file(job_id, status, message=None, output_file=None):
    """Write status information to a file"""
    output_dir = os.environ.get('CONVERSION_STATUS_DIR', 
                              os.path.join(os.getcwd(), 'conversion_status'))
    os.makedirs(output_dir, exist_ok=True)
    
    status_file = os.path.join(output_dir, f"{job_id}_status.json")
    
    # Get existing status if available
    status_data = {}
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                status_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    
    # Update status
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if 'log' not in status_data:
        status_data['log'] = []
    
    status_data['status'] = status
    status_data['last_updated'] = timestamp
    
    if message:
        status_data['message'] = message
        status_data['log'].append({
            'time': timestamp,
            'status': status,
            'message': message
        })
    
    if output_file:
        status_data['output_file'] = output_file
    
    # Write updated status
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2)

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Convert Word templates to Jinja2 format")
    parser.add_argument('--template-path', required=True, help="Path to the source template")
    parser.add_argument('--job-id', default=f"job_{int(time.time())}", help="Job ID for tracking")
    parser.add_argument('--template-type', default='standard', help="Template type (standard, msa, etc)")
    parser.add_argument('--output-dir', help="Custom output directory")
    parser.add_argument('--status-dir', help="Directory for status files")
    
    args = parser.parse_args()
    
    # Set status directory if provided
    if args.status_dir:
        os.environ['CONVERSION_STATUS_DIR'] = args.status_dir
    
    try:
        # Initial cleanup to ensure no Word processes are running
        cleanup_word()
        write_status_file(args.job_id, 'starting', 'Initializing conversion process')
        
        # Initialize COM for main thread
        pythoncom.CoInitialize()
        
        # Convert the template
        write_status_file(args.job_id, 'converting', 'Starting template conversion')
        output_file = convert_template(args.template_path)
        
        # Update status with success
        write_status_file(
            args.job_id, 
            'success', 
            'Conversion completed successfully',
            output_file
        )
        
        log_message(f"Conversion complete. Output file: {output_file}")
        sys.exit(0)
        
    except Exception as e:
        error_message = str(e)
        log_error(f"Conversion failed: {error_message}")
        write_status_file(args.job_id, 'error', f"Conversion failed: {error_message}")
        sys.exit(1)
    finally:
        # Uninitialize COM
        try:
            pythoncom.CoUninitialize()
        except:
            pass

if __name__ == "__main__":
    main()
