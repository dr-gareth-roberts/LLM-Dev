"""
cognitive_metrics.py - Evaluation metrics for reasoning and cognitive capabilities.

This module implements evaluators for assessing reasoning capabilities and cognitive biases,
following British English standards and comprehensive type safety.
"""
import re
import spacy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, cast
from src.evaluation_framework.evaluation_protocols import (
    BaseMetricEvaluator,
    LLMEnvironmentProtocol,
    MetricCategory,
    MetricResult,
    TestCase
)
from src.core.metric_registry import MetricRegistry


@MetricRegistry.register("reasoning", MetricCategory.COGNITIVE)
class ReasoningEvaluator(BaseMetricEvaluator):
    """Evaluates multi-step reasoning and logical consistency."""

    def __init__(self, env: LLMEnvironmentProtocol):
        """Initialise the reasoning evaluator.
        
        Args:
            env: LLM environment for model interaction
        """
        super().__init__(env)
        self.nlp = spacy.load("en_core_web_sm")
        self._load_reasoning_patterns()

    def _load_reasoning_patterns(self) -> None:
        """Load reasoning analysis patterns."""
        self.patterns = {
            "logical_steps": {
                "connectives": [
                    "therefore", "thus", "hence",
                    "because", "since", "as",
                    "consequently", "so", "accordingly"
                ],
                "sequence_markers": [
                    "first", "second", "third",
                    "next", "then", "finally",
                    "lastly", "subsequently"
                ],
                "conditional_markers": [
                    "if", "when", "unless",
                    "provided that", "assuming",
                    "given that", "suppose"
                ]
            },
            "evidence_markers": {
                "citations": [
                    r"according to\s+(\w+)",
                    r"as\s+(\w+)\s+states",
                    r"(\w+)\s+shows that"
                ],
                "data_references": [
                    r"data\s+shows",
                    r"research\s+indicates",
                    r"studies\s+demonstrate",
                    r"evidence\s+suggests"
                ],
                "examples": [
                    r"for example",
                    r"for instance",
                    r"such as",
                    r"specifically"
                ]
            },
            "assumptions": {
                "explicit": [
                    r"assume\s+that",
                    r"assuming\s+that",
                    r"let's\s+say",
                    r"suppose\s+that"
                ],
                "implicit": [
                    r"must\s+be",
                    r"clearly",
                    r"obviously",
                    r"naturally"
                ]
            },
            "counterarguments": {
                "markers": [
                    r"however",
                    r"but",
                    r"although",
                    r"nevertheless",
                    r"despite",
                    r"while"
                ],
                "acknowledgments": [
                    r"one might argue",
                    r"some may say",
                    r"critics suggest",
                    r"an alternative view"
                ]
            }
        }

    async def evaluate(
        self,
        model_id: str,
        test_cases: List[TestCase]
    ) -> MetricResult:
        """Evaluate reasoning capabilities on test cases.
        
        Args:
            model_id: Identifier for the model to evaluate
            test_cases: List of test cases to evaluate
            
        Returns:
            MetricResult: Evaluation results with reasoning scores and details
        """
        details = []
        total_score = 0.0
        
        for test_case in test_cases:
            response = await self.env.get_model_response(model_id, test_case)
            components = self._extract_reasoning_components(response)
            analysis = self._analyse_reasoning_quality(components)
            
            case_score = sum(analysis.values()) / len(analysis)
            total_score += case_score
            
            details.append({
                "prompt": test_case.prompt,
                "response": response,
                "components": components,
                "analysis": analysis,
                "score": case_score
            })
        
        final_score = total_score / len(test_cases)
        return MetricResult(
            score=final_score,
            details=details,
            metadata={
                "model_id": model_id,
                "test_cases": len(test_cases),
                "evaluation_time": datetime.now().isoformat()
            }
        )

    def _extract_reasoning_components(self, text: str) -> Dict[str, List[str]]:
        """Extract reasoning components from text.
        
        Args:
            text: Text to analyse for reasoning components
            
        Returns:
            Dict[str, List[str]]: Extracted reasoning components by category
        """
        doc = self.nlp(text)
        components = {
            "logical_steps": [],
            "evidence": [],
            "assumptions": [],
            "counterarguments": []
        }

        # Extract logical steps
        for sentence in doc.sents:
            # Check for logical connectives
            if any(conn in sentence.text.lower() for conn in self.patterns["logical_steps"]["connectives"]):
                components["logical_steps"].append(sentence.text)

            # Check for sequence markers
            if any(marker in sentence.text.lower() for marker in self.patterns["logical_steps"]["sequence_markers"]):
                components["logical_steps"].append(sentence.text)

        # Extract evidence
        for category, patterns in self.patterns["evidence_markers"].items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                components["evidence"].extend(m.group(0) for m in matches)

        # Extract assumptions
        for category, patterns in self.patterns["assumptions"].items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                components["assumptions"].extend(m.group(0) for m in matches)

        # Extract counterarguments
        for category, patterns in self.patterns["counterarguments"].items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                components["counterarguments"].extend(m.group(0) for m in matches)

        return components

    def _analyse_reasoning_quality(
        self,
        components: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """Analyse quality of reasoning components.
        
        Args:
            components: Dictionary of extracted reasoning components
            
        Returns:
            Dict[str, float]: Quality scores for each reasoning aspect
        """
        scores = {
            "logical_coherence": 0.0,
            "evidence_quality": 0.0,
            "assumption_awareness": 0.0,
            "counterargument_handling": 0.0
        }

        # Analyse logical coherence
        if components["logical_steps"]:
            # Check for presence of sequence markers
            has_sequence = any(
                any(marker in step.lower() for marker in self.patterns["logical_steps"]["sequence_markers"])
                for step in components["logical_steps"]
            )
            # Check for presence of logical connectives
            has_connectives = any(
                any(conn in step.lower() for conn in self.patterns["logical_steps"]["connectives"])
                for step in components["logical_steps"]
            )
            scores["logical_coherence"] = (
                0.7 if has_sequence and has_connectives
                else 0.4 if has_sequence or has_connectives
                else 0.2
            )

        # Analyse evidence quality
        if components["evidence"]:
            # Check for variety of evidence types
            evidence_types = set()
            for evidence in components["evidence"]:
                if any(pattern in evidence.lower() for pattern in self.patterns["evidence_markers"]["citations"]):
                    evidence_types.add("citation")
                if any(pattern in evidence.lower() for pattern in self.patterns["evidence_markers"]["data_references"]):
                    evidence_types.add("data")
                if any(pattern in evidence.lower() for pattern in self.patterns["evidence_markers"]["examples"]):
                    evidence_types.add("example")
            
            scores["evidence_quality"] = len(evidence_types) / 3.0

        # Analyse assumption awareness
        if components["assumptions"]:
            # Check for balance of explicit and implicit assumptions
            explicit_count = sum(
                1 for assumption in components["assumptions"]
                if any(pattern in assumption.lower() for pattern in self.patterns["assumptions"]["explicit"])
            )
            implicit_count = len(components["assumptions"]) - explicit_count
            scores["assumption_awareness"] = min(1.0, (explicit_count + implicit_count * 0.5) / 3.0)

        # Analyse counterargument handling
        if components["counterarguments"]:
            # Check for variety in counterargument handling
            has_markers = any(
                any(marker in ca.lower() for marker in self.patterns["counterarguments"]["markers"])
                for ca in components["counterarguments"]
            )
            has_acknowledgments = any(
                any(ack in ca.lower() for ack in self.patterns["counterarguments"]["acknowledgments"])
                for ca in components["counterarguments"]
            )
            scores["counterargument_handling"] = (
                0.8 if has_markers and has_acknowledgments
                else 0.5 if has_markers or has_acknowledgments
                else 0.2
            )

        return scores


@MetricRegistry.register("instruction_following", MetricCategory.COGNITIVE)
class InstructionFollowingEvaluator(BaseMetricEvaluator):
    """Evaluates how well a model follows instructions and task requirements."""

    def __init__(self, env: LLMEnvironmentProtocol):
        """Initialise the instruction following evaluator.
        
        Args:
            env: LLM environment for model interaction
        """
        super().__init__(env)
        self.nlp = spacy.load("en_core_web_sm")
        self._load_instruction_patterns()

    def _load_instruction_patterns(self) -> None:
        """Load instruction analysis patterns."""
        self.patterns = {
            "task_requirements": {
                "action_verbs": [
                    "analyse", "calculate", "compare",
                    "describe", "evaluate", "explain",
                    "identify", "list", "summarise"
                ],
                "task_markers": [
                    "must", "should", "need to",
                    "required to", "expected to"
                ]
            },
            "constraints": {
                "limitations": [
                    "only", "except", "excluding",
                    "limited to", "at most", "at least"
                ],
                "format_requirements": [
                    "in bullet points", "in paragraphs",
                    "step by step", "in order"
                ]
            },
            "conditions": {
                "temporal": [
                    "before", "after", "during",
                    "while", "until", "when"
                ],
                "logical": [
                    "if", "unless", "provided that",
                    "assuming", "given that"
                ]
            }
        }

    async def evaluate(
        self,
        model_id: str,
        test_cases: List[TestCase]
    ) -> MetricResult:
        """Evaluate instruction following capability on test cases.
        
        Args:
            model_id: Identifier for the model to evaluate
            test_cases: List of test cases to evaluate
            
        Returns:
            MetricResult: Evaluation results with instruction following scores and details
        """
        details = []
        total_score = 0.0
        
        for test_case in test_cases:
            response = await self.env.get_model_response(model_id, test_case)
            instructions = self._extract_instructions(test_case.prompt)
            analysis = self._analyse_task_completion(instructions, response)
            
            case_score = self._calculate_instruction_score(analysis)
            total_score += case_score
            
            details.append({
                "prompt": test_case.prompt,
                "response": response,
                "instructions": instructions,
                "analysis": analysis,
                "score": case_score
            })
        
        final_score = total_score / len(test_cases)
        return MetricResult(
            score=final_score,
            details=details,
            metadata={
                "model_id": model_id,
                "test_cases": len(test_cases),
                "evaluation_time": datetime.now().isoformat()
            }
        )

    def _extract_instructions(self, prompt: str) -> Dict[str, List[str]]:
        """Extract instructions and requirements from prompt.
        
        Args:
            prompt: Input prompt to analyse for instructions
            
        Returns:
            Dict[str, List[str]]: Extracted instructions by category
        """
        doc = self.nlp(prompt)
        instructions = {
            "tasks": [],
            "constraints": [],
            "conditions": []
        }

        for sentence in doc.sents:
            # Extract tasks
            if any(verb in sentence.text.lower() for verb in self.patterns["task_requirements"]["action_verbs"]):
                instructions["tasks"].append(sentence.text)
            if any(marker in sentence.text.lower() for marker in self.patterns["task_requirements"]["task_markers"]):
                instructions["tasks"].append(sentence.text)

            # Extract constraints
            if any(limit in sentence.text.lower() for limit in self.patterns["constraints"]["limitations"]):
                instructions["constraints"].append(sentence.text)
            if any(req in sentence.text.lower() for req in self.patterns["constraints"]["format_requirements"]):
                instructions["constraints"].append(sentence.text)

            # Extract conditions
            if any(temp in sentence.text.lower() for temp in self.patterns["conditions"]["temporal"]):
                instructions["conditions"].append(sentence.text)
            if any(logic in sentence.text.lower() for logic in self.patterns["conditions"]["logical"]):
                instructions["conditions"].append(sentence.text)

        return instructions

    def _analyse_task_completion(
        self,
        instructions: Dict[str, List[str]],
        response: str
    ) -> Dict[str, float]:
        """Analyse how well the response completes required tasks.
        
        Args:
            instructions: Dictionary of extracted instructions
            response: Model's response to analyse
            
        Returns:
            Dict[str, float]: Completion scores for each instruction aspect
        """
        scores = {
            "task_completion": 0.0,
            "constraint_adherence": 0.0,
            "condition_satisfaction": 0.0
        }

        # Analyse task completion
        if instructions["tasks"]:
            completed_tasks = sum(
                1 for task in instructions["tasks"]
                if any(verb in response.lower() for verb in self.patterns["task_requirements"]["action_verbs"])
            )
            scores["task_completion"] = completed_tasks / len(instructions["tasks"])

        # Analyse constraint adherence
        if instructions["constraints"]:
            adhered_constraints = sum(
                1 for constraint in instructions["constraints"]
                if any(limit in response.lower() for limit in self.patterns["constraints"]["limitations"])
                or any(req in response.lower() for req in self.patterns["constraints"]["format_requirements"])
            )
            scores["constraint_adherence"] = adhered_constraints / len(instructions["constraints"])

        # Analyse condition satisfaction
        if instructions["conditions"]:
            satisfied_conditions = sum(
                1 for condition in instructions["conditions"]
                if any(temp in response.lower() for temp in self.patterns["conditions"]["temporal"])
                or any(logic in response.lower() for logic in self.patterns["conditions"]["logical"])
            )
            scores["condition_satisfaction"] = satisfied_conditions / len(instructions["conditions"])

        return scores

    def _calculate_instruction_score(self, analysis: Dict[str, float]) -> float:
        """Calculate final instruction following score.
        
        Args:
            analysis: Dictionary of analysis scores
            
        Returns:
            float: Final weighted score between 0 and 1
        """
        weights = {
            "task_completion": 0.5,
            "constraint_adherence": 0.3,
            "condition_satisfaction": 0.2
        }

        weighted_score = sum(
            score * weights[category]
            for category, score in analysis.items()
        )

        return min(1.0, weighted_score)

@MetricRegistry.register("cognitive_bias", MetricCategory.COGNITIVE)
class CognitiveBiasEvaluator(BaseMetricEvaluator):
    """Evaluates cognitive biases and logical fallacies in model responses."""

    def __init__(self, env: LLMEnvironmentProtocol):
        """Initialise the cognitive bias evaluator.
        
        Args:
            env: LLM environment for model interaction
        """
        super().__init__(env)
        self.nlp = spacy.load("en_core_web_sm")
        self._load_bias_patterns()

    def _load_bias_patterns(self) -> None:
        """Load cognitive bias analysis patterns."""
        self.patterns = {
            "confirmation_bias": {
                "selective_focus": [
                    r"only consider(s|ing)?",
                    r"focus(es|ing)? solely on",
                    r"exclusively look(s|ing)? at"
                ],
                "reinforcement": [
                    r"further prov(es|ing)",
                    r"this confirm(s|ed)",
                    r"as expected",
                    r"clearly shows"
                ]
            },
            "anchoring_bias": {
                "initial_focus": [
                    r"start(s|ing|ed)? with",
                    r"begin(s|ning|an)? from",
                    r"based on (the|this) initial"
                ],
                "adjustment_resistance": [
                    r"remain(s|ed) close to",
                    r"similar to (the|our) first",
                    r"not far from"
                ]
            },
            "availability_bias": {
                "recent_events": [
                    r"recent(ly)? (cases|examples|instances)",
                    r"latest (cases|examples|instances)",
                    r"current trend(s)?"
                ],
                "memorable_instances": [
                    r"notable (cases|examples|instances)",
                    r"well-known (cases|examples|instances)",
                    r"prominent (cases|examples|instances)"
                ]
            },
            "logical_fallacies": {
                "ad_hominem": [
                    r"because they are",
                    r"who are you to",
                    r"their background"
                ],
                "false_dichotomy": [
                    r"either.*or",
                    r"must (be|choose) (between|one of)",
                    r"only two options"
                ],
                "hasty_generalisation": [
                    r"all of them",
                    r"everyone (knows|believes|thinks)",
                    r"nobody would",
                    r"always happens"
                ]
            }
        }

    async def evaluate(
        self,
        model_id: str,
        test_cases: List[TestCase]
    ) -> MetricResult:
        """Evaluate cognitive bias presence in model responses.
        
        Args:
            model_id: Identifier for the model to evaluate
            test_cases: List of test cases to evaluate
            
        Returns:
            MetricResult: Evaluation results with cognitive bias scores and details
        """
        details = []
        total_score = 0.0
        
        for test_case in test_cases:
            response = await self.env.get_model_response(model_id, test_case)
            indicators = self._extract_bias_indicators(response)
            analysis = self._analyse_bias_presence(indicators)
            
            case_score = self._calculate_bias_score(analysis)
            total_score += case_score
            
            details.append({
                "prompt": test_case.prompt,
                "response": response,
                "indicators": indicators,
                "analysis": analysis,
                "score": case_score
            })
        
        final_score = total_score / len(test_cases)
        return MetricResult(
            score=final_score,
            details=details,
            metadata={
                "model_id": model_id,
                "test_cases": len(test_cases),
                "evaluation_time": datetime.now().isoformat()
            }
        )

    def _extract_bias_indicators(self, text: str) -> Dict[str, List[str]]:
        """Extract cognitive bias indicators from text.
        
        Args:
            text: Text to analyse for bias indicators
            
        Returns:
            Dict[str, List[str]]: Extracted bias indicators by category
        """
        indicators = {
            "confirmation_bias": [],
            "anchoring_bias": [],
            "availability_bias": [],
            "logical_fallacies": []
        }

        # Extract confirmation bias indicators
        for category, patterns in self.patterns["confirmation_bias"].items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                indicators["confirmation_bias"].extend(m.group(0) for m in matches)

        # Extract anchoring bias indicators
        for category, patterns in self.patterns["anchoring_bias"].items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                indicators["anchoring_bias"].extend(m.group(0) for m in matches)

        # Extract availability bias indicators
        for category, patterns in self.patterns["availability_bias"].items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                indicators["availability_bias"].extend(m.group(0) for m in matches)

        # Extract logical fallacy indicators
        for category, patterns in self.patterns["logical_fallacies"].items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                indicators["logical_fallacies"].extend(m.group(0) for m in matches)

        return indicators

    def _analyse_bias_presence(
        self,
        indicators: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """Analyse presence and severity of cognitive biases.
        
        Args:
            indicators: Dictionary of extracted bias indicators
            
        Returns:
            Dict[str, float]: Severity scores for each bias type
        """
        scores = {
            "confirmation_bias": 0.0,
            "anchoring_bias": 0.0,
            "availability_bias": 0.0,
            "logical_fallacies": 0.0
        }

        # Analyse confirmation bias
        if indicators["confirmation_bias"]:
            selective_focus = sum(
                1 for indicator in indicators["confirmation_bias"]
                if any(pattern in indicator.lower() for pattern in self.patterns["confirmation_bias"]["selective_focus"])
            )
            reinforcement = sum(
                1 for indicator in indicators["confirmation_bias"]
                if any(pattern in indicator.lower() for pattern in self.patterns["confirmation_bias"]["reinforcement"])
            )
            scores["confirmation_bias"] = min(1.0, (selective_focus + reinforcement) / 4.0)

        # Analyse anchoring bias
        if indicators["anchoring_bias"]:
            initial_focus = sum(
                1 for indicator in indicators["anchoring_bias"]
                if any(pattern in indicator.lower() for pattern in self.patterns["anchoring_bias"]["initial_focus"])
            )
            adjustment_resistance = sum(
                1 for indicator in indicators["anchoring_bias"]
                if any(pattern in indicator.lower() for pattern in self.patterns["anchoring_bias"]["adjustment_resistance"])
            )
            scores["anchoring_bias"] = min(1.0, (initial_focus + adjustment_resistance) / 4.0)

        # Analyse availability bias
        if indicators["availability_bias"]:
            recent_events = sum(
                1 for indicator in indicators["availability_bias"]
                if any(pattern in indicator.lower() for pattern in self.patterns["availability_bias"]["recent_events"])
            )
            memorable_instances = sum(
                1 for indicator in indicators["availability_bias"]
                if any(pattern in indicator.lower() for pattern in self.patterns["availability_bias"]["memorable_instances"])
            )
            scores["availability_bias"] = min(1.0, (recent_events + memorable_instances) / 4.0)

        # Analyse logical fallacies
        if indicators["logical_fallacies"]:
            ad_hominem = sum(
                1 for indicator in indicators["logical_fallacies"]
                if any(pattern in indicator.lower() for pattern in self.patterns["logical_fallacies"]["ad_hominem"])
            )
            false_dichotomy = sum(
                1 for indicator in indicators["logical_fallacies"]
                if any(pattern in indicator.lower() for pattern in self.patterns["logical_fallacies"]["false_dichotomy"])
            )
            hasty_generalisation = sum(
                1 for indicator in indicators["logical_fallacies"]
                if any(pattern in indicator.lower() for pattern in self.patterns["logical_fallacies"]["hasty_generalisation"])
            )
            scores["logical_fallacies"] = min(1.0, (ad_hominem + false_dichotomy + hasty_generalisation) / 6.0)

        return scores

    def _calculate_bias_score(self, analysis: Dict[str, float]) -> float:
        """Calculate final cognitive bias score.
        
        Args:
            analysis: Dictionary of analysis scores
            
        Returns:
            float: Final weighted score between 0 and 1
        """
        weights = {
            "confirmation_bias": 0.3,
            "anchoring_bias": 0.2,
            "availability_bias": 0.2,
            "logical_fallacies": 0.3
        }

        weighted_score = sum(
            score * weights[category]
            for category, score in analysis.items()
        )

        return min(1.0, weighted_score)
