// Prints the user messages Ollama's decision package compiles for each fixture,
// as JSON lines, so systemone.py can be diffed against them byte for byte.
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/decision"
)

func main() {
	var cases []json.RawMessage
	data, _ := os.ReadFile(os.Args[1])
	if err := json.Unmarshal(data, &cases); err != nil {
		panic(err)
	}
	for _, raw := range cases {
		var req decision.Request
		if err := json.Unmarshal(raw, &req); err != nil {
			panic(err)
		}
		req.Model = "x"
		c, err := decision.Compile(req)
		if err != nil {
			panic(err)
		}
		var users []string
		c.Render(func(m []api.Message) (string, error) { users = append(users, m[0].Content); return "", nil })
		out, _ := json.Marshal(users)
		fmt.Println(string(out))
	}
}
