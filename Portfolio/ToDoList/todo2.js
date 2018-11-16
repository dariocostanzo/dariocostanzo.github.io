

var todos = ["buy"];

var input = prompt("What would you like to do?");

while(input !== "quit"){
	//handle input
	
	if(input === "list") {
		listTodo();
	} else if (input === "new") {
		addTodo();		
	}
	else if (input === "delete") {
		deleteTodo();
	}
	//ask again for new input
	input = prompt("What would you like to do?");
	}
	alert("bye!");
	//ask again for new input
	

		function addTodo(){
		//ask for new todo
		var newTodo = prompt("Enter new todo");
		//add to todos array
		todos.push(newTodo);
		alert("Added todo");
	}

	function deleteTodo(){

		//ask for index of todo to be deleted
		var index = prompt("Enter index of todo to  delete");
		//add to todos array
		todos.splice(index, 1); //(item, index to delete)
		alert("Deleted todo");
	}

	function listTodo(){
		console.log("***********");
		todos.forEach(function(todo,i){ // (item, index)
			alert(i + ": " + todo);
		});
		console.log("***********");
	}
