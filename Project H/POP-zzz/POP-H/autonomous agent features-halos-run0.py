class ResearchAgent:
    def __init__(self):
        self.web_connector = SafeWebConnector()

    def research(self, query: str) -> Dict[str, Any]:
        sources = self.web_connector.search(query)
        verified = []
        for source in sources:
            verified.append({
                "source": source,
                "facts": self._verify_facts(source['content'], query)
            })
        return {"results": verified}

    def chain_tasks(self, tasks: List[Dict]) -> Any:
        context = {}
        for task in tasks:
            if task['type'] == 'analyze_code':
                context['analysis'] = code_doctor.analyze_code(task['content'])
            elif task['type'] == 'fix_code':
                context['fixes'] = code_doctor.fix_code(context['analysis'])
            # ... other task types
        return context