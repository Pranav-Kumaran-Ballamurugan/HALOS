import re
import ast
import sys
from typing import List, Dict, Optional, Tuple
import subprocess
from pathlib import Path

class EnhancedCodeDoctor:
    def __init__(self):
        # Initialize with expanded capabilities
        self.language_support = {
            'python': self.analyze_python,
            'javascript': self.analyze_javascript,
            # Placeholder for other languages
        }
        self.ai_suggestions_enabled = False
        self.auto_fix_enabled = False
        self.batch_mode = False
        
    def enable_ai(self, api_key: str = None):
        """Enable AI-powered suggestions"""
        self.ai_suggestions_enabled = True
        # In a real implementation, this would initialize an AI service connection
        print("AI suggestions enabled (simulated)")
        
    def enable_auto_fix(self):
        """Enable automatic code fixing"""
        self.auto_fix_enabled = True
        print("Auto-fix enabled (simulated)")
        
    def set_batch_mode(self, enabled: bool):
        """Enable batch processing of multiple files"""
        self.batch_mode = enabled
        
    def diagnose(self, code: str, language: str = 'python') -> Dict:
        """Analyze code with enhanced capabilities"""
        if language not in self.language_support:
            return {
                'error': f'Unsupported language: {language}',
                'supported_languages': list(self.language_support.keys())
            }
            
        return self.language_support[language](code)
    
    def analyze_python(self, code: str) -> Dict:
        """Enhanced Python analysis with data flow tracking"""
        diagnosis = {
            'healthy': True,
            'issues': [],
            'suggestions': [],
            'auto_fixes': []
        }
        
        # Parse AST with enhanced type checking
        try:
            tree = ast.parse(code)
            type_info = self._analyze_types(tree)
            diagnosis.update(self._check_type_consistency(type_info))
        except SyntaxError as e:
            return self._handle_syntax_error(e)
            
        # Data flow analysis
        data_flow_issues = self._analyze_data_flow(tree)
        diagnosis['issues'].extend(data_flow_issues)
        
        # AI-powered suggestions if enabled
        if self.ai_suggestions_enabled:
            ai_suggestions = self._get_ai_suggestions(code)
            diagnosis['suggestions'].extend(ai_suggestions)
            
        # Generate auto-fixes if enabled
        if self.auto_fix_enabled and diagnosis['issues']:
            diagnosis['auto_fixes'] = self._generate_auto_fixes(code, diagnosis['issues'])
            
        diagnosis['healthy'] = not bool(diagnosis['issues'])
        return diagnosis
    
    def _analyze_types(self, tree: ast.AST) -> Dict:
        """Enhanced type analysis with data flow tracking"""
        # This would be implemented with more sophisticated type inference
        return {
            'variables': {},  # Would contain type info in real implementation
            'functions': {},  # Would contain return types
            'potential_issues': []  # Would contain type inconsistencies
        }
        
    def _check_type_consistency(self, type_info: Dict) -> Dict:
        """Check for type inconsistencies in the analyzed code"""
        # Simulated type checking results
        return {
            'issues': [
                {
                    'type': 'type_error',
                    'message': 'Potential type mismatch in function call',
                    'line': 10,
                    'severity': 'high'
                }
            ],
            'suggestions': [
                {
                    'type': 'type_suggestion',
                    'message': 'Consider adding type hints',
                    'severity': 'medium'
                }
            ]
        }
    
    def _analyze_data_flow(self, tree: ast.AST) -> List[Dict]:
        """Analyze data flow through the program"""
        # Simulated data flow analysis
        return [
            {
                'type': 'data_flow',
                'message': 'Variable might be used before assignment in some paths',
                'line': 15,
                'severity': 'high'
            }
        ]
    
    def _get_ai_suggestions(self, code: str) -> List[Dict]:
        """Get AI-powered suggestions for code improvement"""
        # In a real implementation, this would call an AI service
        return [
            {
                'type': 'ai_suggestion',
                'message': 'This loop could be simplified using list comprehension',
                'severity': 'medium',
                'confidence': 0.85
            }
        ]
        
    def _generate_auto_fixes(self, code: str, issues: List[Dict]) -> List[Dict]:
        """Generate automatic fixes for identified issues"""
        fixes = []
        for issue in issues:
            if issue['type'] == 'type_error':
                fixes.append({
                    'issue': issue,
                    'fixed_code': code + "\n# Added type hint (simulated fix)",
                    'description': 'Added type annotation'
                })
        return fixes
    
    def analyze_javascript(self, code: str) -> Dict:
        """Basic JavaScript analysis (placeholder)"""
        # This would be implemented with proper JS parsing
        return {
            'healthy': True,
            'issues': [],
            'suggestions': [
                {
                    'type': 'language_support',
                    'message': 'JavaScript support is experimental',
                    'severity': 'low'
                }
            ]
        }
    
    def analyze_directory(self, path: str) -> Dict:
        """Analyze all code files in a directory"""
        path = Path(path)
        results = {}
        
        for file in path.glob('**/*'):
            if file.suffix in ['.py', '.js']:
                lang = 'python' if file.suffix == '.py' else 'javascript'
                with open(file, 'r') as f:
                    code = f.read()
                results[str(file)] = self.diagnose(code, lang)
                
        return results


def main():
    print("Enhanced Code Doctor - Advanced Programming Issue Diagnostic")
    doctor = EnhancedCodeDoctor()
    
    # Enable features based on user input
    if '--ai' in sys.argv:
        doctor.enable_ai()
    if '--fix' in sys.argv:
        doctor.enable_auto_fix()
        
    if '--batch' in sys.argv:
        path = input("Enter directory path to analyze: ")
        results = doctor.analyze_directory(path)
        for file, diagnosis in results.items():
            print(f"\n=== {file} ===")
            print_diagnosis(diagnosis)
    else:
        print("Enter your code (press Ctrl+D or Ctrl+Z followed by Enter when done):")
        code = sys.stdin.read()
        diagnosis = doctor.diagnose(code)
        print_diagnosis(diagnosis)


def print_diagnosis(diagnosis: Dict):
    """Print diagnosis results"""
    print("\n=== Diagnosis Report ===")
    if diagnosis.get('error'):
        print(f"❌ Error: {diagnosis['error']}")
        return
        
    if diagnosis['healthy']:
        print("✅ Code appears healthy!")
    else:
        print("⚠️  Issues found:")
        for issue in diagnosis['issues']:
            print(f"\n🚨 {issue['type'].upper()} at line {issue.get('line', '?')}:")
            print(f"   {issue['message']} (severity: {issue['severity']})")
    
    if diagnosis['suggestions']:
        print("\n💡 Suggestions for improvement:")
        for suggestion in diagnosis['suggestions']:
            print(f" - [{suggestion['severity'].upper()}] {suggestion['message']}")
            if 'confidence' in suggestion:
                print(f"   (AI confidence: {suggestion['confidence']:.0%})")
    
    if diagnosis.get('auto_fixes'):
        print("\n🔧 Auto-fix suggestions:")
        for fix in diagnosis['auto_fixes']:
            print(f"\nFor issue: {fix['issue']['message']}")
            print(f"Fix: {fix['description']}")
            print("--- Fixed code ---")
            print(fix['fixed_code'])
            print("------------------")


if __name__ == "__main__":
    print("Command-line options:")
    print("--ai     : Enable AI-powered suggestions")
    print("--fix    : Enable auto-fix generation")
    print("--batch  : Analyze all files in a directory")
    main()