import sys

sys.path.append('/home/roboticslab/Documents/CSSR4Africa_LLM/shared/')

from shared_functions import *

# Global variable to store loaded data items
# data_items = []
search_history = []

def main():
    """Main function for interactive CLI food recommendation system"""
    try:
        print("🍽️  Interactive Upanzi Q&A System")
        print("=" * 50)
        print("Loading 2024 report database...")
        
        # Load food data from file
        # global data_items
        # global search_history
        # data_items = load_json_data("/home/roboticslab/Documents/CSSR4Africa_LLM/RAG/upanzi_program_review_2024_reporting_v2.json")
        # print(f"✅ Loaded {len(data_items)} data items successfully")
        
        # Create and populate search collection
        # collection = create_similarity_search_collection(
        #     "interactive_upanzi_search",
        #     {'description': 'A collection for interactive Upanzi search'}
        # )
        collection = get_similarity_search_collection("interactive_upanzi_search")
        # populate_similarity_collection(collection, data_items)
        
        # Start interactive chatbot
        interactive_upanzi_chatbot(collection)
        
    except Exception as error:
        print(f"❌ Error initializing system: {error}")

def interactive_upanzi_chatbot(collection):
    """Interactive CLI chatbot for Upanzi Network"""
    print("\n" + "="*50)
    print("🤖 INTERACTIVE UPANZI SEARCH CHATBOT")
    print("="*50)
    print("Commands:")
    print("  • Ask anything about Upanzi or make a comment")
    print("  • 'help' - Show available commands")
    print("  • 'quit' or 'exit' - Exit the system")
    print("  • Ctrl+C - Emergency exit")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\n🔍 Ask a question or make a comment: ").strip()
            
            # Handle empty input
            if not user_input:
                print("   Please enter your question or 'help' for commands")
                continue
            
            # Handle exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using the Upanzi Network!")
                print("   Goodbye!")
                break
            
            # Handle help command
            elif user_input.lower() in ['help', 'h']:
                show_help_menu()
            
            elif user_input.lower() in ['history']:
                handle_history_command()
            
            # Handle food search
            else:
                handle_question_or_comment(collection, user_input)
                
        except KeyboardInterrupt:
            print("\n\n👋 System interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing request: {e}")

def show_help_menu():
    """Display help information for users"""
    print("\n📖 HELP MENU")
    print("-" * 30)
    print("Search Examples:")
    print("  • 'Projects' - Find project-related information")
    print("  • 'Funding' - Find out about funding sources")
    print("  • 'Directors' - Find information about directors")
    print("  • 'Objectives' - Find information about objectives")
    print("  • 'Location' - Find information about locations")
    print("\nCommands:")
    print("  • 'help' - Show this help menu")
    print("  • 'quit' - Exit the system")

def handle_question_or_comment(collection, query):
    """Handle question or comment with enhanced display"""
    search_history.append(query)
    print(f"\n🔍 Searching for '{query}'...")
    print("   Please wait...")
    
    # Perform similarity search
    results = perform_similarity_search(collection, query, 5)
    
    if not results:
        print("❌ No matching foods found.")
        print("💡 Try different keywords like:")
        print("   • Cuisine types: 'Italian', 'Thai', 'Mexican'")
        print("   • Ingredients: 'chicken', 'vegetables', 'cheese'")
        print("   • Descriptors: 'spicy', 'sweet', 'healthy'")
        return
    
    # Display results with rich formatting
    print(f"\n✅ Found {len(results)} documents:")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        # Calculate percentage score
        percentage_score = result['similarity_score'] * 100
        
        print(f"\n{i}. 🍽️  {result['doc_id']}")
        print(f"   📊 Match Score: {percentage_score:.1f}%")
        print(f"   🏷️  Section: {result['section']}")
        print(f"   🔥 Content: {result['content']}")
        
        # Add visual separator
        if i < len(results):
            print("   " + "-" * 50)
    
    print("=" * 60)

def handle_history_command():
    """Display user's search history"""
    if not search_history:
        print("📝 No search history available")
        return
    
    print("\n📝 Your Search History:")
    print("-" * 30)
    for i, search in enumerate(search_history[-10:], 1):  # Show last 10
        print(f"{i}. {search}")

if __name__ == "__main__":
    main()