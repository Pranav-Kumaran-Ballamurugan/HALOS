import re
import ast
import sys
from typing import List, Dict, Optional

class CodeDoctor:
    def __init__(self):
        self.symptoms_db = {
            'syntax_error': {
                'description': 'Syntax errors in code',
                'diagnosis': 'The code contains syntax errors that prevent it from being parsed',
                'remedies': [
                    'Check for missing parentheses, brackets, or quotes',
                    'Look for incorrect indentation',
                    'Verify all statements end properly',
                    'Check for invalid variable names or keywords'
                ]
            },
            'indentation_error': {
                'description': 'Inconsistent indentation',
                'diagnosis': 'The code has inconsistent indentation levels',
                'remedies': [
                    'Ensure consistent use of spaces or tabs',
                    'Check that blocks are properly indented',
                    'Verify all function bodies and control structures are indented'
                ]
            },
            'name_error': {
                'description': 'Undefined variable or function',
                'diagnosis': 'A variable or function is being used before definition',
                'remedies': [
                    'Check for typos in variable/function names',
                    'Ensure all variables are defined before use',
                    'Verify imports are correct for external names'
                ]
            },
            'type_error': {
                'description': 'Incompatible operations between types',
                'diagnosis': 'An operation is being performed on incompatible data types',
                'remedies': [
                    'Check variable types with type() or isinstance()',
                    'Verify function parameter types',
                    'Add type conversion where needed'
                ]
            },
            'logic_error': {
                'description': 'Code runs but produces wrong results',
                'diagnosis': 'The code executes without errors but has flawed logic',
                'remedies': [
                    'Add print statements to trace variable values',
                    'Check edge cases in conditional statements',
                    'Verify loop conditions and termination',
                    'Step through code with a debugger'
                ]
            }
        }
        
        self.language = 'python'  # Default language, could be expanded
    
    def diagnose(self, code: str) -> Dict:
        """Analyze code and return diagnostic information"""
        diagnosis = {
            'healthy': True,
            'issues': [],
            'suggestions': []
        }
        
        # Check for syntax errors
        try:
            ast.parse(code)
        except SyntaxError as e:
            diagnosis['healthy'] = False
            issue = self._handle_syntax_error(e)
            diagnosis['issues'].append(issue)
            return diagnosis
        
        # Check for common patterns that might indicate issues
        self._check_code_patterns(code, diagnosis)
        
        # If no issues found, provide general advice
        if diagnosis['healthy']:
            diagnosis['suggestions'].append({
                'type': 'general',
                'message': 'Code appears healthy. Consider adding tests and documentation.',
                'severity': 'low'
            })
        
        return diagnosis
    
    def _handle_syntax_error(self, error: SyntaxError) -> Dict:
        """Process a syntax error into a diagnosis"""
        return {
            'type': 'syntax_error',
            'message': str(error),
            'line': error.lineno,
            'offset': error.offset,
            'suggestions': self.symptoms_db['syntax_error']['remedies'],
            'severity': 'critical'
        }
    
    def _check_code_patterns(self, code: str, diagnosis: Dict):
        """Check for common problematic code patterns"""
        lines = code.split('\n')
        
        # Check for broad exception handling
        for i, line in enumerate(lines):
            if 'except:' in line or 'except Exception:' in line:
                diagnosis['suggestions'].append({
                    'type': 'style',
                    'message': f'Broad exception handling at line {i+1} - consider specific exceptions',
                    'severity': 'medium'
                })
        
        # Check for potential name errors (simple pattern matching)
        variables = set()
        defined_funcs = set()
        
        # Simple pattern to find variable assignments
        var_pattern = r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=[^=]'
        func_def_pattern = r'^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        
        for i, line in enumerate(lines):
            # Track variable assignments
            var_match = re.match(var_pattern, line)
            if var_match:
                variables.add(var_match.group(1))
            
            # Track function definitions
            func_match = re.match(func_def_pattern, line)
            if func_match:
                defined_funcs.add(func_match.group(1))
            
            # Check for variable usage
            words = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[^=][^(]?', line)
            for word in words:
                if (word not in variables and word not in defined_funcs and 
                    not word.iskeyword() and word not in dir(__builtins__)):
                    diagnosis['suggestions'].append({
                        'type': 'potential_name_error',
                        'message': f'Potential undefined variable "{word}" at line {i+1}',
                        'severity': 'high'
                    })
    
    def suggest_fix(self, issue_type: str) -> List[str]:
        """Get suggested fixes for a specific issue type"""
        return self.symptoms_db.get(issue_type, {}).get('remedies', [])
    
    def explain_issue(self, issue_type: str) -> Dict:
        """Get detailed explanation of an issue type"""
        return self.symptoms_db.get(issue_type, {})


def main():
    print("Code Doctor - Programming Issue Diagnostic Tool")
    print("Enter your code (press Ctrl+D or Ctrl+Z followed by Enter when done):")
    
    try:
        code = sys.stdin.read()
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return
    
    doctor = CodeDoctor()
    diagnosis = doctor.diagnose(code)
    
    print("\n=== Diagnosis Report ===")
    if diagnosis['healthy']:
        print("✅ Code appears healthy!")
    else:
        print("⚠️  Issues found:")
        for issue in diagnosis['issues']:
            print(f"\n🚨 {issue['type'].upper()} at line {issue.get('line', '?')}:")
            print(f"   {issue['message']}")
            print("   Suggested fixes:")
            for i, fix in enumerate(issue['suggestions'], 1):
                print(f"   {i}. {fix}")
    
    if diagnosis['suggestions']:
        print("\n💡 Suggestions for improvement:")
        for suggestion in diagnosis['suggestions']:
            print(f" - [{suggestion['severity'].upper()}] {suggestion['message']}")


if __name__ == "__main__":
    main()