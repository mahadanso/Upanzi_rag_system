import sys

sys.path.append('/home/roboticslab/Documents/CSSR4Africa_LLM/shared/')

from shared_functions import *
from typing import List, Dict, Any
import openai


client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key = "sk-no-key-required"
)

def main():
    """Main function for enhanced RAG chatbot system"""
    try:
        print("🤖 Enhanced RAG-Powered Upanzi Chatbot")
        print("   Powered by llama.cpp & ChromaDB")
        print("=" * 55)

        # Get collection for RAG system
        collection = get_similarity_search_collection("interactive_upanzi_search")

        # Start RAG chatbot
        rag_chatbot(collection)
        
    except Exception as error:
        print(f"❌ Error: {error}")

def prepare_context_for_llm(query: str, search_results: List[Dict]) -> str:
    """Prepare structured context from search results for LLM"""
    if not search_results:
        return "No relevant document found in the database."
    
    context_parts = []
    context_parts.append("Based on your query, here are the most relevant documents from our database:")
    context_parts.append("")
    
    for i, result in enumerate(search_results[:3], 1):
        doc_context = []
        doc_context.append(f"Option {i}: {result['section']}")
        doc_context.append(f"  - Content: {result['content']}")
        
        doc_context.append(f"  - Similarity score: {result['similarity_score']*100:.1f}%")
        doc_context.append("")
        
        context_parts.extend(doc_context)
    
    return "\n".join(context_parts)

def generate_llm_rag_response(query: str, search_results: List[Dict], conversation_history: List[str]) -> str:
    """Generate response using llama.cpp with retrieved context"""
    try:
        # Prepare context from search results
        context = prepare_context_for_llm(query, search_results)
        
        # Build the prompt for the LLM
        prompt = f'''You are a helpful upanzi lab assistant. A user is asking questions about the Upanzi, and I've retrieved relevant options from a document database.

User Query: "{query}"

Retrieved Document Information:
{context}

Please provide a helpful, short response that:
1. Acknowledges the user's request
2. Answers the question or comment from the retrieved options
3. Explains why these answers match their request
4. Includes relevant details
5. Uses a friendly, conversational tone
6. Keeps the response concise but informative

Response:'''

        # Generate response using IBM Granite
        generated_response = client.completions.create(
            model="davinci-002",
            prompt=prompt,
            max_tokens=512
        )

        print(f'Generated Response: {type(generated_response)}')

        # Extract the generated text
        if len(generated_response.choices) > 0:
            response_text = generated_response.choices[0].text

            # Clean up the response if needed
            response_text = response_text.strip()
            
            # If response is too short, provide a fallback
            if len(response_text) < 50:
                return generate_fallback_response(query, search_results)
            
            return response_text
        else:
            return generate_fallback_response(query, search_results)
            
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return generate_fallback_response(query, search_results)

def generate_fallback_response(query: str, search_results: List[Dict]) -> str:
    """Generate fallback response when LLM fails"""
    if not search_results:
        return "I couldn't find any documents matching your request. Try rephrasing your question!"
    
    top_result = search_results[0]
    response_parts = []
    
    response_parts.append(f"Based on your request for '{query}', I'd recommend {top_result['section']}.")
    response_parts.append(f"Content: {top_result['content']}.")
    
    if len(search_results) > 1:
        second_choice = search_results[1]
        response_parts.append(f"Another great document would be {second_choice['section']}.")
    
    return " ".join(response_parts)

def rag_chatbot(collection):
    """Enhanced RAG-powered conversational upanzi chatbot"""
    print("\n" + "="*70)
    print("🤖 ENHANCED RAG UPANZI CHATBOT")
    print("   Powered by llama.cpp")
    print("="*70)
    print("💬 Ask me any questions using natural language!")
    print("\nExample queries:")
    print("  • 'What is the Upanzi?'")
    print("  • 'What has Upanzi done so far?'")
    print("  • 'What are the benefits of the Upanzi?'")
    print("  • 'Can you suggest some Upanzi projects?'")
    print("\nCommands:")
    print("  • 'help' - Show detailed help menu")
    print("  • 'compare' - Compare recommendations for two different queries")
    print("  • 'quit' - Exit the chatbot")
    print("-" * 70)
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                print("🤖 Bot: Please ask me a question about Upanzi!")
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n🤖 Bot: Thank you for using the Enhanced RAG Upanzi Chatbot!")
                print("      Hope you found some useful information! 👋")
                break
            
            elif user_input.lower() in ['help', 'h']:
                show_enhanced_rag_help()
            
            elif user_input.lower() in ['compare']:
                handle_enhanced_comparison_mode(collection)
            
            else:
                # Process the food query with enhanced RAG
                ai_response = handle_enhanced_rag_query(collection, user_input, conversation_history)
                conversation_history.append(f'User: {user_input}')
                conversation_history.append(f'Bot: {ai_response}')
                conversation_history.append('-'*20)

                # Keep conversation history manageable
                if len(conversation_history) > 5:
                    conversation_history = conversation_history[-3:]
                
        except KeyboardInterrupt:
            print("\n\n🤖 Bot: Goodbye! Hope you find something useful! 👋")
            break
        except Exception as e:
            print(f"❌ Bot: Sorry, I encountered an error: {e}")

def handle_enhanced_rag_query(collection, query: str, conversation_history: List[str]):
    """Handle user query with enhanced RAG approach"""
    print(f"\n🔍 Searching vector database for: '{query}'...")
    
    # Perform similarity search with more results for better context
    search_results = perform_similarity_search(collection, query, 3)
    
    if not search_results:
        print("🤖 Bot: I couldn't find any documents matching your request.")
        print("      Try rephrasing your question!")
        return
    
    print(f"✅ Found {len(search_results)} relevant matches")
    print("🧠 Generating AI-powered response...")
    
    # Generate enhanced RAG response using IBM Granite
    ai_response = generate_llm_rag_response(query, search_results, conversation_history)
    
    print(f"\n🤖 Bot: {ai_response}")
    
    # Show detailed results for reference
    print(f"\n📊 Search Results Details:")
    print("-" * 45)
    for i, result in enumerate(search_results[:3], 1):
        print(f"{i}. 🍽️  {result['section']}")
        print(f"   📍 {result['content']} | 📈 {result['similarity_score']*100:.1f}% match")
        if i < 3:
            print()

    return ai_response.split('\n')[0]

def handle_enhanced_comparison_mode(collection):
    """Enhanced comparison between two queries using LLM"""
    print("\n🔄 ENHANCED COMPARISON MODE")
    print("   Powered by AI Analysis")
    print("-" * 35)
    
    query1 = input("Enter first query: ").strip()
    query2 = input("Enter second query: ").strip()
    
    if not query1 or not query2:
        print("❌ Please enter both queries for comparison")
        return
    
    print(f"\n🔍 Analyzing '{query1}' vs '{query2}' with AI...")
    
    # Get results for both queries
    results1 = perform_similarity_search(collection, query1, 3)
    results2 = perform_similarity_search(collection, query2, 3)
    
    # Generate AI-powered comparison
    comparison_response = generate_llm_comparison(query1, query2, results1, results2)
    
    print(f"\n🤖 AI Analysis: {comparison_response}")
    
    # Show side-by-side results
    print(f"\n📊 DETAILED COMPARISON")
    print("=" * 60)
    print(f"{'Query 1: ' + query1[:20] + '...' if len(query1) > 20 else 'Query 1: ' + query1:<30} | {'Query 2: ' + query2[:20] + '...' if len(query2) > 20 else 'Query 2: ' + query2}")
    print("-" * 60)
    
    max_results = max(len(results1), len(results2))
    for i in range(min(max_results, 3)):
        left = f"{results1[i]['section']} ({results1[i]['similarity_score']*100:.0f}%)" if i < len(results1) else "---"
        right = f"{results2[i]['section']} ({results2[i]['similarity_score']*100:.0f}%)" if i < len(results2) else "---"
        print(f"{left[:30]:<30} | {right[:30]}")

def generate_llm_comparison(query1: str, query2: str, results1: List[Dict], results2: List[Dict]) -> str:
    """Generate AI-powered comparison between two queries"""
    try:
        context1 = prepare_context_for_llm(query1, results1[:3])
        context2 = prepare_context_for_llm(query2, results2[:3])
        
        comparison_prompt = f'''You are analyzing and comparing two different queries. Please provide a thoughtful comparison.

Query 1: "{query1}"
Top Results for Query 1:
{context1}

Query 2: "{query2}"
Top Results for Query 2:
{context2}

Please provide a short comparison that:
1. Highlights the key differences between these two queries
2. Notes any similarities or overlaps
3. Explains which query might be better for different situations
4. Recommends the best option from each query
5. Keeps the analysis concise but insightful

Comparison:'''

        generated_response = client.completions.create(
            model="davinci-002",
            prompt=prompt,
            max_tokens=512
        )

        if len(generated_response.choices) > 0:
            return generated_response.choices[0].text.strip()
        else:
            return generate_simple_comparison(query1, query2, results1, results2)
            
    except Exception as e:
        return generate_simple_comparison(query1, query2, results1, results2)

def generate_simple_comparison(query1: str, query2: str, results1: List[Dict], results2: List[Dict]) -> str:
    """Simple comparison fallback"""
    if not results1 and not results2:
        return "No results found for either query."
    if not results1:
        return f"Found results for '{query2}' but none for '{query1}'."
    if not results2:
        return f"Found results for '{query1}' but none for '{query2}'."

    return f"For '{query1}', I recommend {results1[0]['section']}. For '{query2}', {results2[0]['section']} would be perfect."

def show_enhanced_rag_help():
    """Display help information for enhanced RAG chatbot"""
    print("\n📖 ENHANCED RAG CHATBOT HELP")
    print("=" * 45)
    print("🧠 This chatbot uses llama.cpp to understand your")
    print("   queries and provide intelligent responses.")
    print("\nHow to get the best responses:")
    print("  • Be specific")
    print("  • Mention preferences")
    print("  • Include context")
    print("  • Ask about benefits")
    print("\nSpecial features:")
    print("  • 🔍 Vector similarity search finds relevant documents")
    print("  • 🧠 AI analysis provides contextual explanations")
    print("  • 📊 Detailed information")
    print("  • 🔄 Smart comparison between different queries")
    print("\nCommands:")
    print("  • 'compare' - AI-powered comparison of two queries")
    print("  • 'help' - Show this help menu")
    print("  • 'quit' - Exit the chatbot")
    print("\nTips for better results:")
    print("  • Use natural language - talk like you would to a friend")
    print("  • Mention preferences")

if __name__ == "__main__":
    main()