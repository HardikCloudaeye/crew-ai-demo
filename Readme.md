# Code Review Demo for Agentic Systems

## Overview
This project demonstrates how to use Cloudaeye's specialized code reviewer to detect bugs and security vulnerabilities in agentic applications. The demo showcases a multi-tenant chatbot built with CrewAI that evolves from a simple web search utility to a RAG-powered assistant, introducing subtle but critical issues during development.
Project Purpose
The primary objective is to showcase Cloudaeye's specialized code reviewer, which has deep knowledge of agentic frameworks like CrewAI, LangChain, and AutoGPT. This reviewer can identify both security vulnerabilities and functional bugs that may not be obvious to developers. The demo serves as:

A practical demonstration of Cloudaeye's agentic code review capabilities
A training tool for developers to understand common pitfalls in agent-based systems
A showcase of how specialized knowledge of agent frameworks enables more effective code review

# Demo Components
1. Multi-tenant Web Chatbot

Initial Version: A simple chatbot that performs web searches for tenant-specific URLs
Authentication: Password-protected access for different organizations
CrewAI Architecture: Uses specialized agents for web research and customer support

2. Evolution & Issues
The demo shows a simulated development journey:

Basic Implementation: Web search functionality with proper tenant isolation
Enhancement: Developer adds RAG capabilities for improved responses
Issues Introduced: During implementation, the developer inadvertently introduces:

```Enter the issues here```

3. Cloudaeye Code Reviewer
The demo concludes with using Cloudaeye's agent-based code reviewer to:

Detect both security vulnerabilities and functional bugs in the implementation
Leverage specialized knowledge of agentic frameworks to identify framework-specific issues
Provide contextual explanation of potential problems and their impact
Generate a comprehensive technical report prioritizing issues by severity


Getting Started

Clone the repository
Install dependencies with pip install -r requirements.txt
Configure tenant information in tenant_info.json
Run the application with python app.py
Access the web interface at http://localhost:5000
