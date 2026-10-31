# Person and agent chat

This is the record of the Cursor session for the CSE606 labyrinth, written in the order the AI disclosure statement asks for it. It is not a shortened rewrite.

Tool: Cursor, using Grok.

## 1. Code generation and modification

**User:** Complete this assignment, just like FurniturePlanner.

**Agent:** FurniturePlanner is a vanilla WebGL page. The labyrinth PDF is the 3D assignment: modular walls and floors, a start and an end, three to five third-party surface meshes, top, world, and player views, matrix transforms, quaternion rotation with a virtual trackball, mouse transforms, and keyboard view changes. I generate `index.html`, `main.js`, `style.css`, the JSDoc setup, and embed the Burkardt PLY files for the airplane, teapot, shark, ant, and scissors.

**User:** Read the assignment. Isn't it 3D?

**Agent:** Yes. Top view is the flat design mode. World view and player view are perspective. The first screenshot was top view, and the maze sat in the corner because the orthographic box was in world coordinates after `lookAt` had already centered the camera. That projection is corrected in `main.js`.

**User:** Camera controls suck in the egocentric movement view.

**Agent:** Play mode was multiplying each turn onto the camera's current quaternion, so pitch leaked into yaw and the horizon rolled. Right-drag also ran the trackball on the player. The view quaternion is now rebuilt from yaw, pitch, and roll. Clicking the maze captures the mouse. W and S walk, A and D turn, F and G strafe, Q and E look up and down, R levels the view.

**User:** Make it a little aesthetic, and the touches.

**Agent:** I change the panel, the stone floor with grout, the wall caps, a short fog in the 3D views, and a crosshair in play view.

What the form's checklist maps to in this chat:

- HTML boilerplate: yes, the page shell and shaders in `index.html`
- JavaScript event listeners: yes, mouse, wheel, keys, resize
- WebGL boilerplate: yes, shader compile, program link, buffers
- Math helpers: yes, including quaternion rebuild, trackball, and ray tests
- GLSL: yes, the vertex and fragment shaders, including the later fog and two-light shade

Modification level: the model wrote those files. They were then corrected in the same chat for the top-view projection, the player camera, and the visual pass. They have not been rewritten by hand outside this chat.

## 2. 3D modeling and assets

**User:** (no Blender or image-model request)

**Agent:** No Blender add-on, no generated textures, no denoising. The obstacles are John Burkardt's public ASCII PLY meshes, parsed in `parsePLY`. Walls, floors, the player marker, and the end pyramid are boxes and a pyramid built in code.

Checklist: None for AI modeling tools.

## 3. Report writing and documentation

**Agent:** JSDoc comments were written with the source. A separate written report was not finished in this chat. If a report is drafted later from these notes, say so on the form. Do not describe that text as only a grammar pass over writing you already had.

Checklist so far: inline JSDoc, yes. Report outline, drafting, and proofreading, not done yet.

## 4. Academic integrity

The form's certification says the core rendering loop, shader logic, and matrix pipeline were engineered by the student, and that the agent was used only for supplemental boilerplate, troubleshooting, and editing.

This chat does not support that sentence. The rendering loop, the shaders, and the transform path were generated here. Troubleshooting after that was also done here: the orthographic camera, the first-person look, and the visual pass.

Sign section 4 only after you have checked that code and can stand behind it. The transcript link should point at this file, not at a rewritten conversation.
